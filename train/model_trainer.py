# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch model trainer

from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch
import matplotlib.pyplot as plt
from plot_listen.plot import plot_training_loss
from tqdm import tqdm, tqdm_notebook
import os
import datetime
from util import delete_dir, torch_to_numpy, print_globals, simple_name_of_obj, create_dir, count_params
from ui import list_selection_menu
from eval.training_evaluator import TrainingEvaluator
from IPython.display import clear_output
import soundfile as sf
import sys
import shutil
import ipdb
from globals import *


class ModelTrainer:
    """
    Class for a generic model trainer.
    """
    def __init__(self, model, train_dataset, test_dataset,
                device=DEVICE, batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS,
                train_shuffle=TRAIN_SHUFFLE, optimizer=OPTIMIZER, 
                learning_rate=LEARNING_RATE,
                lr_step_size=LEARNING_RATE_STEP_SIZE,
                lr_decay=LEARNING_RATE_DECAY,
                evaluator=None,
                output_path=OUTPUT_PATH):
        """
        Initialize model trainer.

        Parameters
        ----------
        model : pytorch model
            model to be trained
        train_dataset : pytorch dataset
            data to train with
        test_dataset : pytorch dataset
            data to test with
        device : gpu or cpu device
            device to train on
        batch_size : int 
            number of items per batch
        num_workers : int 
            number of workers to use parallel processing for dataloading
        train_shuffle : bool
            if True training data will be shuffled
        optimizer : torch optim
            torch optimizer to use to update model weights
        learning_rate : float
            learning rate parameter for the optimizer
        lr_step_size : int
            step size (in epochs) for the learning rate scheduler
        lr_decay : float
            amount to multiply learning rate by every epoch
        evaluator : TrainingEvaluator
            evaluator object. If none will create a default one from this trainer
        output_path : string
            path to output folder
                 
        """
        self.model = model.to(device)
        self.device = device

        self.batch_size = batch_size
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.train_dataset_name = train_dataset.name
        self.test_dataset_name = test_dataset.name

        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=lr_step_size, gamma=lr_decay)

        self.loss_function = model.loss_function
 
        self.evaluator = evaluator # None by default, will be set up in experiment setup if need be

        self.output_path = output_path

        self.epochs_trained = 0

        self.training_loss_values = []
        self.validation_loss_values = []

        # function for getting a list of data by accessing a list of keys stored in model.config
        # for instance model.config['input'] = ["f0", "loudness"] for training a basic mono synth
        # might be useful in util so that evalutor can use?
        self.data_from_config_key = lambda D, x: [D[key].to(self.device) for key in self.model.config[x]]
    
    def load_checkpoint(self, output_path=OUTPUT_PATH, experiment_name=None, checkpoint_name=None, checkpoint_path=None, load_latest=True):
        """
        Loads model weights and training checkpoint.

        Parameters
        ----------
        output_path : str
            path to model outputs
        experiment_name : str
            name of the experiment (and its dir)
        checkpoint_name : str
            name of the checkpoint (for instance "26 epochs")
        load_latest : bool
            if true, loads the latest checkpoint
            else, prompts user to select  
        """

        if checkpoint_path is not None:
            load_path = checkpoint_path
        else:
            load_path = output_path
            if experiment_name is None:
                # To Do: sort by date and show n top?
                # Filter experiments to include only those with matching model name
                list_avail_exp_names = [exp_name for exp_name in os.listdir(output_path) if self.model.name in exp_name]
                experiment_name = list_selection_menu(list_avail_exp_names)   
            load_path = os.path.join(load_path, experiment_name, "checkpoints")
            if checkpoint_name is None:
                list_avail_cp_names = os.listdir(load_path)
                list_avail_cp_names.sort(reverse=True, key=lambda x: float(x.split()[0])) # cp names are like "125 epochs"...sort by number, largest first
                if load_latest:
                    checkpoint_name = list_avail_cp_names[0]
                    print(f"loading checkpoint: {checkpoint_name}")
                else:
                    checkpoint_name = list_selection_menu(list_avail_cp_names)   
            load_path = os.path.join(load_path, checkpoint_name, "model_checkpoint.pt")

        loaded_checkpoint = torch.load(load_path)

        # save the experiment path, for continued training with consolidated output
        self.experiment_path = os.path.join(output_path, experiment_name)

        self.epochs_trained = loaded_checkpoint['epochs_trained']
        self.model.load_state_dict(loaded_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])
        self.training_loss_values = loaded_checkpoint['training_loss_values']
        self.validation_loss_values = loaded_checkpoint['validation_loss_values']

        print("checkpoint loaded.")

    
    def experiment_setup(self, consolidate_output=False):
        """
        Assigns experiment info and creates a new directory to save output checkpoint + results.
        Saves off txt file of global params and In code

        Parameters
        ----------
        consolidate_output : bool
            Deafult is False, creates a new directory for this experiment
            If True, uses directory from loaded model (if possible)

        """
        # get the date and time for logging purposes.
        now = datetime.datetime.utcnow()+datetime.timedelta(hours=TIME_ZONE_HOURS)
        date_time_str = now.strftime("%Y-%m-%d %I:%M%p")

        if not consolidate_output:

            # create model name + path
            experiment_description = input("Enter a brief experiment description: ")
            self.experiment_name = f"{date_time_str} {self.model.name} : {experiment_description}"
            self.experiment_path = os.path.join(self.output_path, self.experiment_name)

            create_dir(self.experiment_path)

        # save off txt of summary including model, data, and globals.
        orig_stdout = sys.stdout # save the standard output so we can go back after writing to file
        summary_filename = f"summary_{date_time_str}.txt"
        with open(os.path.join(self.experiment_path, summary_filename), 'w') as f:
            sys.stdout = f
            print("Model\n----------\n", self.model)
            print("\n\nTrainable Params\n----------", count_params(self.model))
            print("\n\nLoss Function\n----------\n", simple_name_of_obj(self.loss_function))
            print("\n\nTrain Dataset\n----------\n", self.train_dataset_name)
            print("\n\nTest Dataset\n----------\n", self.test_dataset_name)
            print("\n\nGlobals\n----------")
            print_globals()
        sys.stdout = orig_stdout

        # create self.evaluator if need be
        if self.evaluator is None:
            self.evaluator = TrainingEvaluator(trainer=self)
    
    def delete_prev_checkpoint(self, prev_checkpoint_epochs, epochs_per_checkpoint_retained=EPOCHS_PER_CHECKPOINT_RETAINED):
        """
        Deletes the previous checkpoint folder unless it is an index that should be retained.

        Parameters
        ----------
        prev_checkpoint_epochs : int
            number of epochs in the last checkpoint
        epochs_per_checkpoint_retained : int
            checkpoints that have epoch counts that are multiples of this number are not deleted      
        """
        if prev_checkpoint_epochs % epochs_per_checkpoint_retained == 0:
            # return before deleting
            return

        prev_checkpoint_name = f"{prev_checkpoint_epochs} epochs" if prev_checkpoint_epochs != 1 else f"{prev_checkpoint_epochs} epoch"
        prev_checkpoint_path = os.path.join(self.experiment_path, "checkpoints", prev_checkpoint_name)
        delete_dir(prev_checkpoint_path)


    def train_epoch(self):
        """
        Trains for one epoch of the dataset. (Seeing every item once)
        """

        # make sure the model is set to train
        self.model.train()

        self.train_loss = 0.0

        for i, data_dict in tqdm_notebook(enumerate(self.train_dataloader), total=len(self.train_dataloader)):

            # get the inputs and labels using the keys as part of the model's config dict
            inputs = self.data_from_config_key(data_dict, "inputs")
            labels = self.data_from_config_key(data_dict, "labels")

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward pass
            output_dict = self.model(*inputs)
            outputs = self.data_from_config_key(output_dict, "outputs")

            # compute loss
            loss = self.loss_function(*outputs, *labels) # * to unpack the list
            if USE_REGULARIZATION_LOSS:
                loss += self.model.compute_regularization_loss()

            # backward pass
            loss.backward()

            # step optimizer
            self.optimizer.step()

            # keep track of the epoch's loss
            self.train_loss += loss.item()
        
        # step scheduler at end of epoch
        self.scheduler.step()        
        
        # divide train_loss by number of batches to compare against validation loss
        self.train_loss =  self.train_loss / len(self.train_dataloader)
        
        # save this epoch's training loss
        self.training_loss_values.append(self.train_loss)

    def valid_epoch(self):
        """
        Runs all validation data through the model and computes validation loss
        """

        # make sure the model is set to eval for validation
        self.model.eval()

        self.valid_loss = 0.0

        # loop through batches
        for i, data_dict in tqdm_notebook(enumerate(self.test_dataloader), total=len(self.test_dataloader)):

            # get the inputs and labels using the keys as part of the model's config dict
            inputs = self.data_from_config_key(data_dict, "inputs")
            labels = self.data_from_config_key(data_dict, "labels")

            # forward pass
            output_dict = self.model(*inputs)
            outputs = self.data_from_config_key(output_dict, "outputs")

            # compute loss
            loss = self.loss_function(*outputs, *labels) # * to unpack the list
            if USE_REGULARIZATION_LOSS:
                loss += self.model.compute_regularization_loss()

            # keep track of the epoch's loss
            self.valid_loss += loss.item()     
        
        # divide valid_loss by number of batches so I can compare against train loss
        self.valid_loss =  self.valid_loss / len(self.test_dataloader)
        
        # save this epoch's validation loss
        self.validation_loss_values.append(self.valid_loss) 

    def print_losses(self):
        """
        Prints the curring training and validation loss
        """
        print(f"training loss: {self.train_loss}")
        print(f"validation loss: {self.valid_loss}")
        
    
    def train_model(self, num_epochs=NUM_EPOCHS, 
                    epochs_per_checkpoint=EPOCHS_PER_CHECKPOINT, 
                    epochs_per_checkpoint_retained=EPOCHS_PER_CHECKPOINT_RETAINED,
                    consolidate_output=False):
        """
        Trains model for multiple epochs

        Parameters
        ----------
        num_epochs : int
            number of epochs to train for
        epochs_per_checkopoint : int
            how frequently to save a checkpoint
        epochs_per_checkopoint_retained : int
            will delete check points unless they are multiples of this value
        consolidate_output : bool
            Deafult is False, creates a new directory for this experiment
            If True, uses directory from loaded model (if possible)
        """
        # create experiment name and save path
        self.experiment_setup(consolidate_output)

        prev_checkpoint = -1 # to store the # epochs in the last checkpoint
        for i in range(num_epochs):
            print(f"\nepoch: {self.epochs_trained}")
            self.train_epoch()
            self.valid_epoch()
            clear_output(wait=True) # clears the output cell (so that it doesn't become super long)
            self.print_losses()
            self.epochs_trained += 1
            # save checkpoint after the first epoch, then all multiples of the epochs_per_checkpoint 
            if self.epochs_trained == 1 or self.epochs_trained % epochs_per_checkpoint == 0:
                self.checkpoint_model()
                self.delete_prev_checkpoint(prev_checkpoint, epochs_per_checkpoint_retained) # will not delete if multiple of epochs_per_checkpoint_retained
                prev_checkpoint = self.epochs_trained

    def checkpoint_model(self):
        """
        Saves a checkpoint with model weights + training_loss + any other evaluation
        """

        # create checkpoint dir
        checkpoint_name = f"{self.epochs_trained} epochs" if self.epochs_trained != 1 else f"{self.epochs_trained} epoch"
        checkpoint_path = os.path.join(self.experiment_path, "checkpoints", checkpoint_name)
        create_dir(checkpoint_path)
        
        # save model weights, trainer params
        model_checkpoint_file = os.path.join(checkpoint_path, "model_checkpoint.pt")
        torch.save({"epochs_trained" : self.epochs_trained,
                    "model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict(),
                    "scheduler_state_dict" : self.scheduler.state_dict(),
                    "training_loss_values" : self.training_loss_values,
                    "validation_loss_values" : self.validation_loss_values},
                    model_checkpoint_file)

        # evaluate mode
        self.model.eval()

        # perform any evaulations specified by model
        self.evaluator.checkpoint_eval(checkpoint_path)


        
