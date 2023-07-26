# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for decoding a single string conditioning and embedding into synth params

import torch.nn as nn
import torch.nn.functional as F
from ddsp.mlp import MLP
from midi_synth.midi_util import midi_to_hz
from midi_synth.residual_gru import ResidualGRU
import math
from globals import *

class MonoNetwork(nn.Module):
    """
    Torch nn decoder for taking in a single string's midi conditioning and the string label and outputting synth params.
    Based on the monophonic network from: https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf.
    But the size and layout is more like the original DDSP paper.
    """
    def __init__(self, 
                mlp_num_units=MLP_NUM_UNITS, 
                mlp_num_layers=MLP_NUM_LAYERS, 
                mlp_activation=MLP_ACTIVATION,
                gru_num_units=GRU_NUM_UNITS,
                use_string_embedding=MIDI_SYNTH_USE_STRING_EMBEDDING,
                use_context_net=USE_CONTEXT_NET,
                string_embedding_num=STRING_EMBEDDING_NUM,
                string_embedding_size=STRING_EMBEDDING_SIZE,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS,
                use_mfcc_input=MIDI_SYNTH_USE_MFCC_INPUT,
                learn_pitch_adjustment=MIDI_SYNTH_LEARN_PITCH_ADJUSTMENT):
        """
        Initialize DDSP decoder module.


        Parameters
        ----------
        mlp_num_units : int
            number of units in each mlp layer
        mlp_num_layers : int
            number of layers in each mlp
        mlp_activation : torch.nn module
            activation to apply in mlp layers
        gru_num_units : int
            number of units in the gru's hidden layer
        use_string_embedding : bool
            if True, use string index as input to mono network
        string_embedding_num : int
            number of possible strings (6 for now, but should it be 6 x num_guitars eventually?)
        string_embedding_size : int
            size of the string embedding vector
        num_oscillators : int
            number of oscillators to get harm distribution for (corresponds to number of oscillators in synth)
        num_filter_coeffs : int
            number of filter coefficients to output for use in filtered noise synth
        use_mfcc_input : bool
            if true, then synth expects audio as input so that mfccs can be calculated
        learn_pitch_adjustment : bool
            if True, the pitch conditioning can be adjusted by a network
        """
        super().__init__()

        # create string embedding layer, if required
        self.use_string_embedding = use_string_embedding
        if self.use_string_embedding:
            self.string_emb_layer = nn.Embedding(string_embedding_num, string_embedding_size)
        else:
            print("Not using string embedding in mono network.")
            string_embedding_size = 0 # if not using string embedding then dont need to factor in its size


        self.num_inputs = 2 # not including string embedding (2 for pitch/vel. 3 for pitch/vel/mfcc)

        # create input multi-layer perceptrons
        self.mlp_pitch = MLP(num_units=mlp_num_units,
                            num_layers=mlp_num_layers,
                            activation=mlp_activation)
                            
        self.mlp_vel = MLP(num_units=mlp_num_units,
                            num_layers=mlp_num_layers,
                            activation=mlp_activation)
        
        self.use_context_net=use_context_net
        if self.use_context_net:
            context_size = 32 # add to globals
        else:
            context_size = 0

        # optionally include mfcc as an input
        self.use_mfcc_input = use_mfcc_input
        if self.use_mfcc_input:
            self.num_inputs = 3
            self.mlp_mfcc = MLP(input_size=N_MFCC,
                                num_units=mlp_num_units,
                                num_layers=mlp_num_layers,
                                activation=mlp_activation)

        # create gated recurrent unit
        gru_input_size = self.num_inputs * mlp_num_units + string_embedding_size + context_size
        self.gru = nn.GRU(input_size=gru_input_size,
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

        # create pitch adjustment residual gru if needed
        self.learn_pitch_adjustment = learn_pitch_adjustment
        if self.learn_pitch_adjustment:
            self.pitch_adjustment_gru = ResidualGRU()


    def forward(self, conditioning, string_idx=None, mfcc=None, context=None):
        """
        Process string conditioning and idx outputs synth params.

        Parameters
        ----------
        conditioning: tensor (batch, frames, pitch/vel=2)
            pitch and onset velocity labels for a single string
            assumed to be normalized in [0,1]
        string_idx: IntTensor : (batch)
            string index (0 for E-string, 1 for A, etc.) for each item in batch

        Returns
        ----------
            p['H'] : tensor (batch, frames, filter_len)
                output noise filter coefficients
            p['overall_amplitude'] : tensor (batch, frames)
                overall amplitude values
            p['harm_distr'] : tensor (batch, # harmonics, frames)
                harmonic distribution (sum to 1) envelope tensor
            
        """

        # extract parts of normalized conditioning
        pitch = conditioning[:,:,0] # (batch, frames)
        vel = conditioning[:,:,1] # (batch, frames)

        # get string embedding and expand along frames dimension
        if self.use_string_embedding:
            string_emb = self.string_emb_layer(string_idx) # (batch, embedding_size)
            num_frames = conditioning.shape[1]
            string_emb = string_emb[:, None, :].expand(-1, num_frames, -1) # (batch, frames, embedding_size)...-1 leaves dim alone

        # expand to give channel dimension of 1 at the end and put inputs into mlps
        latent_pitch = self.mlp_pitch(pitch[:, :, None])
        latent_vel = self.mlp_vel(vel[:, :, None])
        if self.use_mfcc_input:
            latent_mfcc = self.mlp_mfcc(mfcc)

        # concatenate and feed desired latent/embedded inputs into gru
        if self.use_mfcc_input and self.use_string_embedding:
            gru_input = torch.cat([latent_pitch, latent_vel, latent_mfcc, string_emb], -1)
        elif self.use_mfcc_input:
            gru_input = torch.cat([latent_pitch, latent_vel, latent_mfcc], -1)
        elif self.use_string_embedding:
            gru_input = torch.cat([latent_pitch, latent_vel, string_emb], -1)
        else:
            gru_input = torch.cat([latent_pitch, latent_vel], -1)

        # concatenate on context
        if self.use_context_net:
            gru_input = torch.cat([gru_input, context], -1)

        gru_output = self.gru(gru_input)[0] # index at 0 to get just the output, not the final hidden state
      
        # concatenate gru output with outputs from f0 and loudness mlp
        if self.use_mfcc_input:
            final_mlp_input = torch.cat([gru_output, latent_pitch, latent_vel, latent_mfcc], -1)
        else:
            final_mlp_input = torch.cat([gru_output, latent_pitch, latent_vel], -1)
        
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

        # add f0 (converted from midi pitch) to params dict so that it can be used by the oscillator
        # if learn pitch adjustment, adjust the pitch
        if self.learn_pitch_adjustment:
            pitch = self.pitch_adjustment_gru(pitch)
        midi_pitch = pitch * MIDI_NORM # scale from [0,1] to [0,127]
        f0 = midi_to_hz(midi_pitch)
        f0[torch.where(midi_pitch <= 0.0)] = 0 # midi pitch values less than or equal 0.0 should become 0 Hz


        # create output dict
        p = {}
        
        # store results in dict and return
        p['overall_amplitude'] = overall_amplitude
        p['harm_distr'] = harm_distr
        p['H'] = H
        p['f0'] = f0

        return p
    
