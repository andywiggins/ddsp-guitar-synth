# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch dataset class for loading saved-off single string data

from globals import *
import os
import numpy as np
from torch.utils.data import Dataset
from ddsp.sweetcocoa.sweetcocoa_loudness_extractor import LoudnessExtractor
import ui
from util import torch_to_numpy
import csv
from tqdm import tqdm_notebook

class SingleStringDataset(Dataset):
    """
    Torch dataset class for loading saved-off single string data.
    """
    def __init__(self, name=None, datasets_path=DATASETS_PATH, dtype=DEFAULT_NP_DTYPE, loudness_stats=None):
        """
        Initialize parameters for loading guitarset.

        Parameters
        ----------
        name : string
            Name of dataset to load. If None, give dropdown of options from data_set path?
        datasets_path : string
            Path to datasets
        dtype : numpy data type
            numpy data type to load data as
        loudness_stats : dict
            loudness_stats["mean"] : mean loudness value
            loudness_stats["std"] : standard deviation loudness value 
            if loudness_stats is None, will compute mean and std from this data

        """
        if name==None:
            list_of_avail_datasets = os.listdir(datasets_path)
            name = ui.list_selection_menu(list_of_avail_datasets)

        load_path = os.path.join(datasets_path, name)
        x = np.load(load_path)

        self.audio = x['audio'].astype(dtype)
        self.f0 = x['f0'].astype(dtype)
        self.voicing = x['voicing'].astype(dtype)
        self.loudness = x['loudness'].astype(dtype)

        self.len = len(self.audio)

        self.name = name

        if loudness_stats is None:
            self.loudness_stats = {}
            self.loudness_stats["mean"] = np.mean(self.loudness)
            self.loudness_stats["std"] = np.std(self.loudness)
        else:
            self.loudness_stats = loudness_stats

        self.adjust_loudness()

        if USE_SWEETCOCOA_LOUDNESS_EXTRACTOR:
            loudness_ex = LoudnessExtractor(sr=SR, frame_length=HOP_LENGTH, device=DEVICE)
            audio_tensor = torch.tensor(self.audio, device=DEVICE)
            self.loudness = torch_to_numpy(loudness_ex(audio_tensor))
            print("sweetcocoa loudness extracted.")
            print("shape:", self.loudness.shape)    

    def __len__(self):
        """
        Return length of the dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """
        Return data for the item at idx.

        Parameters
        ----------
        idx : sliceobj
            Indices to access data from

        Returns
        ----------
        item_dict : dict of np arrays
        	Dict of the chosen items' audio, f0, voicing, and loudness
        """
        item_dict = {'audio': self.audio[idx],
                    'f0': self.f0[idx],
                    'voicing': self.voicing[idx],
                    'loudness': self.loudness[idx]}

        return item_dict

    def adjust_loudness(self):
        """
        Adjusts the loudness to be centered at zero with std of 1, based on self.loudness stats
        """
        mean = self.loudness_stats["mean"]
        std = self.loudness_stats["std"]
        self.loudness = (self.loudness - mean) / std