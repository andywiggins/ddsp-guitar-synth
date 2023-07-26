# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for decoding a single string conditioning and embedding into synth params

import torch.nn as nn
import torch.nn.functional as F
from ddsp.mlp import MLP
from midi_synth.midi_util import midi_to_hz
from midi_synth.residual_gru import ResidualGRU
import math
from globals import *

class MonoNetworkSmall(nn.Module):
    """
    Torch nn decoder for taking in a single string's midi conditioning and the string label and outputting synth params.
    Based on the monophonic network from: https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf.
    """
    def __init__(self, 
                linear_1_size=MONO_NET_SM_LINEAR_1_SIZE,
                linear_2_size=MONO_NET_SM_LINEAR_2_SIZE,
                gru_num_units=MONO_NET_SM_GRU_NUM_UNITS,
                use_string_embedding=MIDI_SYNTH_USE_STRING_EMBEDDING,
                use_context_net=USE_CONTEXT_NET,
                context_size=CONTEXT_SIZE,
                string_embedding_num=STRING_EMBEDDING_NUM,
                string_embedding_size=STRING_EMBEDDING_SIZE,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS,
                use_mfcc_input=MIDI_SYNTH_USE_MFCC_INPUT):
        """
        Initialize DDSP decoder module.


        Parameters
        ----------
        linear_1_size : int
            size of 1st dense layer
        linear_2_size : int
            size of 2nd dense layer
        gru_num_units : int
            unit size of gru
        use_context_net : bool
            if True, use context as an input
        context_size : int
            size of the context signal
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

        # set input/output feature size
        self.num_inputs = 2 # for pitch and velocity

        # create string embedding layer, if required
        self.use_string_embedding = use_string_embedding
        if self.use_string_embedding:
            self.string_emb_layer = nn.Embedding(string_embedding_num, string_embedding_size)
            self.num_inputs += string_embedding_size

        # create context net, if required        
        self.use_context_net=use_context_net
        if self.use_context_net:
            self.num_inputs += context_size 

        # optionally include mfcc as an input
        self.use_mfcc_input = use_mfcc_input
        if self.use_mfcc_input:
            self.num_inputs += N_MFCC

        # create layers
        self.linear1 = nn.Linear(self.num_inputs, linear_1_size)
        self.leakyReLU1 = nn.LeakyReLU()
        self.gru = nn.GRU(input_size=linear_1_size,
                        hidden_size=gru_num_units,
                        batch_first=True)
        self.linear2 = nn.Linear(gru_num_units, linear_2_size)
        self.leakyReLU2 = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(linear_2_size)

        # create output linear layers
        self.dense_amplitudes = nn.Linear(linear_2_size, num_oscillators + 1) # plus 1 for overall amplitude
        self.dense_filter_coeffs = nn.Linear(linear_2_size, num_filter_coeffs)

        # modified sigmoid output activiation
        # as in DDSP paper
        self.modified_sigmoid = lambda x: 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7


    def forward(self, conditioning, string_idx=None, mfcc=None, context=None):
        """
        Process string conditioning and idx outputs synth params.

        Parameters
        ----------
        conditioning : tensor (batch, frames, pitch/vel=2)
            pitch and onset velocity labels for a single string
            assumed to be normalized in [0,1]
        string_idx : IntTensor : (batch)
            string index (0 for E-string, 1 for A, etc.) for each item in batch
        context : tensor (batch, frames, context_size)
            context signal

        Returns
        ----------
            p['H'] : tensor (batch, frames, filter_len)
                output noise filter coefficients
            p['overall_amplitude'] : tensor (batch, frames)
                overall amplitude values
            p['harm_distr'] : tensor (batch, # harmonics, frames)
                harmonic distribution (sum to 1) envelope tensor
            p['f0] : tensor (batch, frames)
                f0, from input conditioning
            
        """
        # set up inputs
        inputs = conditioning

        # get string embedding and expand along frames dimension
        if self.use_string_embedding:
            string_emb = self.string_emb_layer(string_idx) # (batch, embedding_size)
            num_frames = conditioning.shape[1]
            string_emb = string_emb[:, None, :].expand(-1, num_frames, -1) # (batch, frames, embedding_size)...-1 leaves dim alone
            inputs = torch.cat([inputs, string_emb], -1)

        # if using context, concatenate on context signal
        if self.use_context_net:
            inputs = torch.cat([inputs, context], -1)

        # if using mfcc, concatenate onto inputs
        if self.use_mfcc_input:
            inputs = torch.cat([inputs, mfcc], -1)

        # feed inputs thru
        linear1_out = self.linear1(inputs)
        leaky1_out = self.leakyReLU1(linear1_out)
        gru_out = self.gru(leaky1_out)[0]
        linear2_out = self.linear2(gru_out)
        leaky2_out = self.leakyReLU2(linear2_out)
        layer_norm_out = self.layer_norm(leaky2_out)

        # get amplitude tensor and filter coeffs H with final dense layers + modified sigmoid
        amplitude_tensor =  self.modified_sigmoid(self.dense_amplitudes(layer_norm_out))
        H = self.modified_sigmoid(self.dense_filter_coeffs(layer_norm_out))
        
        # treat first amplitude as overall, rest as harmonic distribution
        # force harm_distr to sum to one
        overall_amplitude = amplitude_tensor[..., 0]
        harm_distr =  F.softmax(amplitude_tensor[..., 1:], dim=-1)# make distribution sum to 1
        harm_distr = torch.permute(harm_distr, (0,2,1))  # make it be (batch, num_oscillators, frames)

        # add f0 (converted from midi pitch) to params dict so that it can be used by the oscillator
        midi_pitch = conditioning[:,:,0] * MIDI_NORM # scale from [0,1] to [0,127]
        f0 = midi_to_hz(midi_pitch)
        f0[torch.where(midi_pitch <= 0.0)] = 0 # midi pitch values less than or equal 0.0 should become 0 Hz

        # create a string mask from the midi pitch conditioning
        # use this to zero out all param values when the midi conditioning is zero
        if MASK_STRING_DATA_WITH_CONDITIONING:
            # generate mask using f0 (which is zero/non zero tracking string activity)
            string_mask = torch.zeros_like(f0) # (batch, frames), values [0,1]
            string_mask[torch.where(f0 > 0.0)] = 1.0 # set one for all locations where f0 is nonzero

            # widen nonzero areas of string mask
            # sum together shifted copies and clip to [0,1]
            if EXTEND_STRING_MASK:
                left_extend = STRING_MASK_LEFT_EXTEND_FRAMES
                right_extend = STRING_MASK_RIGHT_EXTEND_FRAMES
                string_mask_left_shift = F.pad(string_mask[:,left_extend:], (0,left_extend))
                if right_extend > 0: # only try to right shift if nonzero
                    string_mask_right_shift = F.pad(string_mask[:,:(right_extend * -1)], (right_extend,0))
                else: # if no right shift, don't need to slice/pad
                    string_mask_right_shift = string_mask
                string_mask_sum = string_mask_left_shift + string_mask_right_shift
                string_mask = torch.clip(string_mask_sum, 0, 1)


            # apply mask to all other params
            overall_amplitude = overall_amplitude * string_mask # (batch, frames)
            harm_distr = harm_distr * string_mask[:, None, :] # (batch, # harmonics, frames)
            H = H * string_mask[:, :, None] # (batch, frames, filter_len)

        # create output dict
        p = {}
        
        # store results in dict and return
        p['overall_amplitude'] = overall_amplitude
        p['harm_distr'] = harm_distr
        p['H'] = H
        p['f0'] = f0

        return p