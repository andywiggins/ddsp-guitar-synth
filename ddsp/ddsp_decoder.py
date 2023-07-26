# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for decoding labels into synth params

import torch.nn as nn
import torch.nn.functional as F
from ddsp.mlp import MLP
import math
from globals import *

class DDSPDecoder(nn.Module):
    """
    Torch nn decoder for taking in f0 and loudness labels (and eventually timbre?) and output synth params.
    Based on the decoder used in the original DDSP paper.
    """
    def __init__(self,  
                use_timbre=False, 
                mlp_num_units=MLP_NUM_UNITS, 
                mlp_num_layers=MLP_NUM_LAYERS, 
                mlp_activation=MLP_ACTIVATION,
                gru_num_units=GRU_NUM_UNITS,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS):
        """
        Initialize DDSP decoder module.


        Parameters
        ----------
        use_timbre : bool
		    whether or not timbre is provided as an input
        mlp_num_units : int
            number of units in each mlp layer
        mlp_num_layers : int
            number of layers in each mlp
        mlp_activation : torch.nn module
            activation to apply in mlp layers
        gru_num_units : int
            number of units in the gru's hidden layer
        num_oscillators : int
            number of oscillators to get harm distribution for (corresponds to number of oscillators in synth)
        filter_coeffs : int
            number of filter coefficients to output for use in filtered noise synth
        """
        super().__init__()

        self.use_timbre = use_timbre
        if self.use_timbre:
            print("Timbre not yet implemented in DDSP decoder")
            self.num_inputs = 3
            exit()
        else:
            self.num_inputs = 2


        # create input multi-layer perceptrons
        self.mlp_f0 = MLP(num_units=mlp_num_units,
                            num_layers=mlp_num_layers,
                            activation=mlp_activation)
                            
        self.mlp_loudness = MLP(num_units=mlp_num_units,
                            num_layers=mlp_num_layers,
                            activation=mlp_activation)

        # create gated recurrent unit
        self.gru = nn.GRU(input_size=(self.num_inputs * mlp_num_units),
                        hidden_size=gru_num_units,
                        batch_first=True,
                        bidirectional=True)

        # create final multilayer perceptron
        self.final_mlp = MLP(input_size=(2 * gru_num_units + self.num_inputs * mlp_num_units),   # times 2 for gru because the bidirectional gru has twice as many outputs?
                            num_units=mlp_num_units,
                            num_layers=mlp_num_layers,
                            activation=mlp_activation)

        # create output dense layers
        self.dense_amplitudes = nn.Linear(mlp_num_units, num_oscillators + 1) # plus 1 for overall amplitude
        self.dense_filter_coeffs = nn.Linear(mlp_num_units, num_filter_coeffs)

        # modified sigmoid output activiation
        # as in DDSP paper
        self.modified_sigmoid = lambda x: 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7


    def forward(self, f0, loudness):
        """
        Process input f0, loudness (+ timbre eventually), outputs synth params.

        Parameters
        ----------
        f0: tensor (batch, frames)
            f0 labels
        loudness: tensor (batch, frames)
            f0 labels

        Returns
        ----------
        p : dict of synth params
            p['f0'] : tensor (batch, frames)
                f0 labels
            p['H'] : tensor (batch, frames, filter_len)
                output noise filter coefficients
            p['overall_amplitude'] : tensor (batch, frames)
                overall amplitude values
            p['harm_distr'] : tensor (batch, # harmonics, frames)
                harmonic distribution (sum to 1) envelope tensor
            
        """

        # create output dict and store f0, which gets passed along
        p = {}
        p['f0'] = f0

        # expand to give channel dimension of 1 at the end and put inputs into mlps
        latent_f0 = self.mlp_f0(f0[:, :, None])
        latent_loudness = self.mlp_loudness(loudness[:, :, None])

        # concatenate and feed into gru
        gru_output = self.gru(torch.cat([latent_f0, latent_loudness], -1))[0] # index at 0 to get just the output, not the final hidden state
        
        # concatenate gru output with outputs from f0 and loudness mlp
        final_mlp_input = torch.cat([gru_output, latent_f0, latent_loudness], -1)
        
        # pass into final mlp
        final_mlp_output = self.final_mlp(final_mlp_input)

        # get amplitude tensor and filter coeffs H with final dense layers + modified sigmoid
        amplitude_tensor =  self.modified_sigmoid(self.dense_amplitudes(final_mlp_output))
        H = self.modified_sigmoid(self.dense_filter_coeffs(final_mlp_output))
        
        # treat first amplitude as overall, rest as harmonic distribution
        # force harm_distr to sum to one
        overall_amplitude = amplitude_tensor[..., 0]
        harm_distr =  F.softmax(amplitude_tensor[..., 1:], dim=-1)# make distribution sum to 1
        harm_distr = torch.permute(harm_distr, (0,2,1))  # make it be (batch, num_oscillators, frames)
        
        # store results in dict and return
        p['overall_amplitude'] = overall_amplitude
        p['harm_distr'] = harm_distr
        p['H'] = H

        return p