import torch
import torch.fft
import torch.nn.functional as F
from torch import nn, einsum, xlogy_
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# Basically vision transformer, ViT that accepts MFCC + SpecAug. Refer to:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, pre_norm=True, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        P_Norm = PreNorm if pre_norm else PostNorm

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                P_Norm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                P_Norm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.self_attn = Attention(d_model, heads = num_heads, dim_head = d_ff, dropout = dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, attn_output_weights = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class FilterAttention(nn.Module):
    def __init__(self, output_size=40, d_model=116, num_heads=2, num_layers=1, d_ff=128, max_seq_length=98, dropout=0.1):
        super(FilterAttention, self).__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 5), bias=False),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 3), bias=False),
            nn.GELU(),
        )

        self.fc_mean = nn.Sequential(
            nn.Flatten(),
            nn.Linear(116, output_size),
            nn.Sigmoid(),
        )
        self.fc_std = nn.Sequential(
            nn.Flatten(),
            nn.Linear(116, output_size),
            nn.Sigmoid(),
        )

    def forward(self, src):
        conv_output = self.conv(src)
        enc_output = conv_output.squeeze()
        # enc_output = src.squeeze()
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)
        enc_output = enc_output.mean(dim=1)

        mean = self.fc_mean(enc_output)
        std = self.fc_std(enc_output)
        return mean, std


class KWT(nn.Module):
    def __init__(self, input_res, patch_res, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., pre_norm = True, adaptive_model=False, **kwargs):
        super().__init__()
        
        num_patches = int(input_res[0]/patch_res[0] * input_res[1]/patch_res[1])
        
        patch_dim = channels * patch_res[0] * patch_res[1]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_res[0], p2 = patch_res[1]),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, pre_norm, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.adaptive_model = adaptive_model
        self.filter_net = FilterAttention()
        self.n_fft = 480
        self.device = 'cuda'
        self.dct_filters = self.dct(dct_filter_num=40, filter_len=40)
    
    def dct(self, dct_filter_num, filter_len):
        basis = torch.empty((dct_filter_num, filter_len), dtype=torch.float32)
        basis[0, :] = 1.0 / torch.sqrt(torch.tensor(filter_len, dtype=torch.float32))

        samples = torch.arange(1, 2 * filter_len, 2) * (torch.tensor(np.pi) / (2.0 * torch.tensor(filter_len, dtype=torch.float32)))

        for i in range(1, dct_filter_num):
            basis[i, :] = torch.cos(i * samples) * torch.sqrt(2.0 / torch.tensor(filter_len, dtype=torch.float32))

        return basis.to(self.device)

    def create_filters(self, bs, fm, bw, donorm=False):
        x = torch.zeros((bs, self.n_fft // 2 + 1), device=self.device) # device=device
        x[:, :] = torch.arange(0, self.n_fft // 2 + 1).to(self.device)
        fm = (self.n_fft / 2 + 1) * fm
        bw = 4 * bw + 0.2

        n_filters = fm.shape[-1]
        l_filter = self.n_fft // 2 + 1


        filters = torch.zeros((bs, 1, n_filters, l_filter), device=self.device) # device=device
        for nf in range(n_filters):
            if donorm:
                filters[:, 0, nf, :] = (torch.exp(-(x.T - fm[:, nf]) ** 2 / (2 * bw[:, nf] ** 2)) * (torch.exp(-fm[:, nf] * 0.01)) ).T
            else:
                filters[:, 0, nf, :] = (torch.exp(-(x.T - fm[:, nf]) ** 2 / (2 * bw[:, nf] ** 2))).T

        # enorm = 2.0 / (mel_freqs[2:mel_filter_num + 2] - mel_freqs[:mel_filter_num])
        # filters *= enorm[:, np.newaxis]
        return filters
        
    def forward(self, x):
        if self.adaptive_model:
            fm, bw = self.filter_net(x)
            bs = fm.shape[0]
            filters = self.create_filters(bs, fm, bw, donorm=False)
            x_filtered = torch.matmul(filters, torch.transpose(x, 2, 3))
            x_log = 10.0 * torch.log10(x_filtered + 1e-9)
            x = torch.matmul(self.dct_filters, x_log)

        x = self.to_patch_embedding(x)
        
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


def kwt_from_name(model_name: str):

    print('here')

    models = {
        "kwt-1": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 256,
            "dim": 64,
            "heads": 1,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        },

        "kwt-2": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 512,
            "dim": 128,
            "heads": 2,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        },

        "kwt-3": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 768,
            "dim": 192,
            "heads": 3,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        }
    }

    assert model_name in models.keys(), f"Unsupported model_name {model_name}; must be one of {list(models.keys())}"

    return KWT(**models[model_name])
