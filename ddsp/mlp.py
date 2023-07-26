# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn module for multilayer perceptron

import torch.nn as nn
from globals import *

from globals import MLP_NUM_LAYERS

class MLP(nn.Module):
    """
    Multilayer perceptron module used in the DDSP decoder.
    Each layer is: Dense, Layer Norm, RELU

    """
    def __init__(self, input_size=MLP_INPUT_SIZE, num_units=MLP_NUM_UNITS, num_layers=MLP_NUM_LAYERS, activation=MLP_ACTIVATION):
        """
        Initialize filtered noise module.

        Parameters
        ----------
        input_size : int
            size of input
        num_units : int
            number of nodes per layer
        num_layers : int
            number of layers
        activation : nn module
            activation to apply to end of each layer, (defaults to ReLU)
        """
        super().__init__()

        self.num_units = num_units
        self.num_layers = num_layers

        # set up first layer
        self.module_list = [
            nn.Linear(input_size, num_units),
            nn.LayerNorm(num_units),
            activation()
        ]

        # all the other layers
        for i in range(num_layers - 1):
            self.module_list.append(nn.Linear(num_units, num_units))
            self.module_list.append(nn.LayerNorm(num_units))
            self.module_list.append(activation())

        # unpack module list as input to sequential
        self.model = nn.Sequential(*self.module_list)

    def forward(self, x):
        """
        Pass input x through the multilayer perceptron

        Parameters
        ----------
        x : tensor (*, input_size)		    

        Returns
        ----------
        output : tensor (*, output_size)
        	Output processed by the network

        """
        return self.model(x)