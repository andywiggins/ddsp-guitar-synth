# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch dataset class for loading saved-off single string data

from globals import *
import os
import numpy as np
from torch.utils.data import Dataset
import ui
from util import torch_to_numpy
import csv
from tqdm import tqdm_notebook

class HexStringDataset(Dataset):
    """
    Torch dataset class for loading saved-off hex string data.
    """
    def __init__(self, name=None, datasets_path=DATASETS_PATH, dtype=DEFAULT_NP_DTYPE):
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

        self.dataset = {}

        for key in x.keys():
            key_s = key[:-4] # shorten the key by removing the "_arr" from the end of the key string
            self.dataset[key_s] = x[key].astype(dtype)

        self.len = len(self.dataset['mix_audio']) # get the length of the last item to be the length of the dataset

        self.name = name

        # Should I center the loudnesses here?

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
        	Dict of the chosen items' data
        """
        
        # load overall data
        item_dict = {}

        # load overall data
        for key in self.dataset:
            item_dict[key] = self.dataset[key][idx]

        return item_dict

