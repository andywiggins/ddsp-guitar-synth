# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch context network for the midi synth

import torch
import torch.nn as nn
from globals import *

class ContextNetwork(nn.Module):
    """
    Torch nn module for a context network that processes global conditioning to be fed into each mono network call.
    Simliar to that describe in the DDSP piano paper.

    """
    def __init__(self, input_size=12, context_size=CONTEXT_SIZE):
        """
        Initialize context net module.

        Parameters
        ----------
        input_size : int
            default=12, thats pitch/vel for 6 strings
        context_size : int
            size of output contex signal default=32
        """
        super().__init__()

        # linear - 32
        self.linear1 = nn.Linear(input_size, 32)
        # leaky relu
        self.leakyReLU = nn.LeakyReLU()
        # gru 64
        self.gru = nn.GRU(input_size=32,
                        hidden_size=64,
                        batch_first=True)
        # normalize 
        self.layer_norm = nn.LayerNorm(64)
        # 2nd linear - context size
        self.linear2 = nn.Linear(64, context_size)


    def forward(self, conditioning):
        """
        Description

        Parameters
        ----------
        conditioning: tensor (batch, frames, num_strings=6, pitch/vel=2)
            pitch and onset velocity labels across 6 strings

        Returns
        ----------
        context : tensor (batch, frames, context_size)
        """

        # flatten last two dims
        x = conditioning.flatten(start_dim=2, end_dim=3) # (batch, frames, 12)

        # linear 1
        x = self.linear1(x) 
        # leaky relu
        x = self.leakyReLU(x)
        # gru
        x = self.gru(x)[0]
        # normalize 
        x = self.layer_norm(x)
        # linear 2
        context = self.linear2(x)

        return context



        




