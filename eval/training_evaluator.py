# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch model evaluator

from torch.utils.data import DataLoader
import torch.optim
import torch
from tqdm import tqdm, tqdm_notebook
import os
import datetime
from util import create_dir, torch_to_numpy
from ui import list_selection_menu
from save_load import save_batch_of_audio_from_data_dict, save_training_loss_plot, save_losses_plot
import matplotlib.pyplot as plt
import soundfile as sf
from globals import *


class TrainingEvaluator:
    """
    Class for evaluating a model.
    """
    def __init__(self, trainer=None, model=None):
        """
        Initialize model evaluator. If given a trainer, will by default use the model and dataloaders associated with the trainer. if trainer is None, a model and train and test data_loaders should be provided

        Parameters
        ----------
        trainer : model_trainer
            trainer object being used
        model : pytorch model
            model to be trained
        """
        if trainer is not None:
            self.trainer = trainer
            self.model = trainer.model
        else:
            self.trainer = None
            self.model = model

    def send_batch_thru_model(self, data_loader, num_items=ITEMS_PER_CHECKPOINT_EVAL):
        """
        Sends a batch of data thru the model. Num items can be provided to do smaller than a batch.

        Parameters
        ----------
        data_loader : DataLoader
            Loader to get batch of data from
            The dictionaries at key 'inputs' will be unpacked and plugged into model
        num_items : int (< batch size)
            Number of items to crop batch to if we want fewer items than in a batch

        Returns
        ----------
        batch_dict : dict
            dictionary of batch to be input into the model
        batch_output_dict : dict
            dictionary of model outputs

        """
        #train items
        batch_dict = next(iter(data_loader))
        batch_dict = {key: batch_dict[key][:num_items] for key in batch_dict} # crop each array in the dict to the desired # of items
        batch_inputs = self.trainer.data_from_config_key(batch_dict, "inputs")
        batch_output_dict = self.model(*batch_inputs)
        return batch_dict, batch_output_dict
    
    def get_and_save_audios(self, data_loader, save_path, dir_name, dict_key='audio'):
        """
        Saves input (target) and output audios from batch a given DataLoader.

        Parameters
        ----------
        dataloader : torch DataLoader
            to draw data from
        save_path : string
            path to save to
        dir_name : string
            directory name to create and save within
        dict_key : string
            key to access data dicts to get audio
        """
        input, output = self.send_batch_thru_model(data_loader)
        curr_path = os.path.join(save_path, dir_name)
        create_dir(curr_path)
        target = self.model.config['labels'][0] # the label in the model config is the target ex: mic_audio for hex_synth with reverb
        save_batch_of_audio_from_data_dict(input, "target", curr_path, target)
        save_batch_of_audio_from_data_dict(output, "resynth", curr_path, dict_key)

    def checkpoint_eval(self, save_path=None):
        """
        Evaluates a ddsp mono synth according to the provided eval_dict, saving off any necessary files.

        Parameters
        ----------
        save_path : string
            path to save to
            if none, will not save
        """
        eval_dict = self.model.eval_dict

        plt.ioff() # turn interactive mode off (don't display the plots)

        if save_path is None:
            print("save path not provided to evaluator. not evaluating")
            return

        if 'training_loss_plot' in eval_dict and eval_dict['training_loss_plot']:
            save_training_loss_plot(self.trainer, save_path)
        
        if 'losses_plot' in eval_dict and eval_dict['losses_plot']:
            save_losses_plot(self.trainer, save_path)
        
        if 'audio' in eval_dict and eval_dict['audio']:
            self.get_and_save_audios(self.trainer.train_dataloader, save_path, "train_audio")
            self.get_and_save_audios(self.trainer.test_dataloader, save_path, "test_audio")

        if 'harmonic_audio' in eval_dict and eval_dict['harmonic_audio']: 
            self.get_and_save_audios(self.trainer.train_dataloader, save_path, "train_harmonic", dict_key="harmonic")
            self.get_and_save_audios(self.trainer.test_dataloader, save_path, "test_harmonic", dict_key="harmonic")

        if 'noise_audio' in eval_dict and eval_dict['noise_audio']: 
            self.get_and_save_audios(self.trainer.train_dataloader, save_path, "train_noise", dict_key="noise")
            self.get_and_save_audios(self.trainer.test_dataloader, save_path, "test_noise", dict_key="noise")

        if 'reverb_audio' in eval_dict and eval_dict['reverb_audio']: 
            self.get_and_save_audios(self.trainer.train_dataloader, save_path, "train_reverb_audio", dict_key="reverb_audio")
            self.get_and_save_audios(self.trainer.test_dataloader, save_path, "test_reverb_audio", dict_key="reverb_audio")

        plt.ion() # turn interactive mode back on after evaluating
            

