import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import functools
import librosa
import glob
import os
from tqdm import tqdm
import multiprocessing as mp
import json
from scipy.signal import get_window
import scipy.fftpack as fft

from utils.augment import time_shift, resample, spec_augment
from audiomentations import AddBackgroundNoise


def get_train_val_test_split(root: str, val_file: str, test_file: str, noises_snr=[]):
    """Creates train, val, and test split according to provided val and test files.

    Args:
        root (str): Path to base directory of the dataset.
        val_file (str): Path to file containing list of validation data files.
        test_file (str): Path to file containing list of test data files.
    
    Returns:
        train_list (list): List of paths to training data items.
        val_list (list): List of paths to validation data items.
        test_list (list): List of paths to test data items.
        label_map (dict): Mapping of indices to label classes.
    """
    
    ####################
    # Labels
    ####################

    label_list = [label for label in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, label)) and label[0] != "_"]
    label_map = {idx: label for idx, label in enumerate(label_list)}

    ###################
    # Split
    ###################

    all_files_set = set()
    for label in label_list:
        all_files_set.update(set(glob.glob(os.path.join(root, label, "*.wav"))))
    
    with open(val_file, "r") as f:
        val_files_set = set(map(lambda a: os.path.join(root, a), f.read().rstrip("\n").split("\n")))
    
    with open(test_file, "r") as f:
        test_files_set = set(map(lambda a: os.path.join(root, a), f.read().rstrip("\n").split("\n"))) 
    
    assert len(val_files_set.intersection(test_files_set)) == 0, "Sanity check: No files should be common between val and test."
    
    all_files_set -= val_files_set
    all_files_set -= test_files_set
    
    train_list, val_list, test_list = list(all_files_set), list(val_files_set), list(test_files_set)

    temp_train = []
    temp_val = []
    temp_test = []
    for noise_snr in noises_snr:
        ns = '__' + noise_snr + '__'
        for x in train_list:
            temp_train.append(x[:-4] + ns + '.wav')

        for x in val_list:
            temp_val.append(x[:-4] + ns + '.wav')

        for x in test_list:
            temp_test.append(x[:-4] + ns + '.wav')

    train_list += temp_train
    val_list += temp_val
    test_list += temp_test
        
    
    print(f"Number of training samples: {len(train_list)}")
    print(f"Number of validation samples: {len(val_list)}")
    print(f"Number of test samples: {len(test_list)}")

    return train_list, val_list, test_list, label_map


class GoogleSpeechDataset(Dataset):
    """Dataset wrapper for Google Speech Commands V2."""
    
    def __init__(
        self, 
        data_list: list, 
        audio_settings: dict, 
        label_map: dict = None, 
        aug_settings: dict = None, 
        cache: int = 0,
        model_type: int = 0,
    ):
        super().__init__()

        self.audio_settings = audio_settings
        self.aug_settings = aug_settings
        self.cache = cache
        self.n_fft = 480
        self.model_type = model_type

        if cache:
            print("Caching dataset into memory.")
            self.data_list = init_cache(data_list, audio_settings["sr"], cache, audio_settings, model_type=self.model_type)
        else:
            self.data_list = data_list
        # self.data_list = data_list
            
        # labels: if no label map is provided, will not load labels. (Use for inference)
        if label_map is not None:
            self.label_list = []
            label_2_idx = {v: int(k) for k, v in label_map.items()}
            for path in data_list:
                self.label_list.append(label_2_idx[path.split("/")[-2]])
        else:
            self.label_list = None
        

        if aug_settings is not None:
            if "bg_noise" in self.aug_settings:
                self.bg_adder = AddBackgroundNoise(sounds_path=aug_settings["bg_noise"]["bg_folder"])


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        if self.cache:
            x = self.data_list[idx]
        else:
            x = librosa.load(self.data_list[idx], sr=self.audio_settings["sr"])[0]
        
        x = self.transform(x)

        if self.label_list is not None:
            label = self.label_list[idx]
            return x, label
        else:
            return x


    def transform(self, x):
        """Applies necessary preprocessing to audio.

        Args:
            x (np.ndarray) - Input waveform; array of shape (n_samples, ).
        
        Returns:
            x (torch.FloatTensor) - MFCC matrix of shape (n_mfcc, T).
        """

        sr = self.audio_settings["sr"]

        ###################
        # Waveform 
        ###################

        if self.cache < 2:
            if self.aug_settings is not None:
                if "bg_noise" in self.aug_settings:
                    x = self.bg_adder(samples=x, sample_rate=sr)

                if "time_shift" in self.aug_settings:
                    x = time_shift(x, sr, **self.aug_settings["time_shift"])

                if "resample" in self.aug_settings:
                    x, _ = resample(x, sr, **self.aug_settings["resample"])
            
            x = librosa.util.fix_length(x, size=sr)

            ###################
            # Spectrogram
            ###################
        
            x = librosa.feature.melspectrogram(y=x, **self.audio_settings)        
            x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=self.audio_settings["n_mels"])\
        
        
        if self.model_type == 0 and self.aug_settings is not None:
            if "spec_aug" in self.aug_settings:
                x = spec_augment(x, **self.aug_settings["spec_aug"])

        x = torch.from_numpy(x).float().unsqueeze(0)
        return x


def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def frame_audio(audio, n_fft=480):
    # audio = np.pad(audio, int(n_fft / 2), mode='reflect')
    frame_len = 160
    frame_num = int((len(audio) - n_fft) / frame_len) + 1
    frames = np.zeros((frame_num, n_fft))
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+n_fft]
    return frames


def cache_item_loader(path: str, sr: int, cache_level: int, audio_settings: dict, window=None, model_type=0) -> np.ndarray:
    x = librosa.load(path, sr=sr)[0]
    if cache_level == 2:
        if model_type == 0:
            x = librosa.util.fix_length(x, size=sr)
            x = librosa.feature.melspectrogram(y=x, **audio_settings)        
            x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=audio_settings["n_mels"])
        else:
            n_fft = 480
            x = librosa.util.fix_length(x, size=sr)
            audio = normalize_audio(x)
            audio_framed = frame_audio(audio)
            audio_win = audio_framed * window
                # (3) fft
            audio_winT = np.transpose(audio_win)
            audio_fft = np.empty((int(1 + n_fft // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
            for n in range(audio_fft.shape[1]):
                audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
            audio_fft = np.transpose(audio_fft)
            x = np.square(np.abs(audio_fft))
    return x


def init_cache(data_list: list, sr: int, cache_level: int, audio_settings: dict, n_cache_workers: int = 4, model_type: int = 0) -> list:
    """Loads entire dataset into memory for later use.

    Args:
        data_list (list): List of data items.
        sr (int): Sampling rate.
        cache_level (int): Cache levels, one of (1, 2), caching wavs and spectrograms respectively.
        n_cache_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        cache (list): List of data items.
    """

    cache = []
    window = get_window('hann', 480, fftbins=True)

    loader_fn = functools.partial(cache_item_loader, sr=sr, cache_level=cache_level, audio_settings=audio_settings, window=window, model_type=model_type)

    pool = mp.Pool(n_cache_workers)

    for audio in tqdm(pool.imap(func=loader_fn, iterable=data_list), total=len(data_list)):
        cache.append(audio)
    
    pool.close()
    pool.join()

    return cache


def get_loader(data_list, config, train=True):
    """Creates dataloaders for training, validation and testing.

    Args:
        config (dict): Dict containing various settings for the training run.
        train (bool): Training or evaluation mode.
        
    Returns:
        dataloader (DataLoader): DataLoader wrapper for training/validation/test data.
    """
    
    with open(config["label_map"], "r") as f:
        label_map = json.load(f)

    model_type = 0
    if config["hparams"]["model]["adaptive_model"]:
        model_type = 1
    
    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=label_map,
        audio_settings=config["hparams"]["audio"],
        aug_settings=config["hparams"]["augment"] if train else None,
        cache=config["exp"]["cache"],
        model_type=model_type,
    )

    print(f'length of dataset: {dataset.__len__()}')

    dataloader = DataLoader(
        dataset,
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
        shuffle=True if train else False
    )

    return dataloader

    
