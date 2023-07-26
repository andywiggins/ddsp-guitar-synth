# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn module for residual GRU

import torch.nn as nn
from globals import *

class ResidualGRU(nn.Module):
    """
    Simple GRU with residual connection for 1-channel sequences

    """
    def __init__(self):
        """
        Initialize residual gru module.
        """
        super().__init__()

        # create gated recurrent unit
        self.gru = nn.GRU(input_size=1,
                        hidden_size=1,
                        batch_first=True)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Pass input x through the residual gru

        Parameters
        ----------
        x : tensor (batch, frames)		    

        Returns
        ----------
        output : tensor (batch, frames)
        	Output processed by the network

        """
        gru_input = x[..., None] # (batch, frames, 1)

        gru_output = self.gru(gru_input)[0] # [0] to get output, not final hidden state, shape (batch, frames, 1)

        gru_output = gru_output[:,:,0] # shape: (batch, frames)

        residual_output = x + self.tanh(gru_output)

        return residual_output