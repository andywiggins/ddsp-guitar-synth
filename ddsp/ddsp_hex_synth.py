# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch ddsp hex synth module includes 6 ddsp mono synths

from ddsp.ddsp_decoder import DDSPDecoder
from ddsp.ddsp_mono_synth import DDSPMonoSynth
from ddsp.harmonic_oscillator import HarmonicOscillator
from ddsp.filtered_noise import FilteredNoise
from ddsp.trainable_reverb import TrainableReverb
from ddsp.multi_scale_spectral_loss import multi_scale_spectral_loss
import torch.nn as nn
import save_load
from globals import *

class DDSPHexSynth(nn.Module):
    """
    Torch nn module incorporating the ddsp decoder and the harmonic + noise synthesizers.
    """
    def __init__(self,
                sr=SR, 
                hop_length=HOP_LENGTH,
                use_timbre=False, 
                mlp_num_units=MLP_NUM_UNITS, 
                mlp_num_layers=MLP_NUM_LAYERS, 
                mlp_activation=MLP_ACTIVATION,
                gru_num_units=GRU_NUM_UNITS,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS,
                eval_dict=DDSP_HEX_SYNTH_EVAL_DICT,
                mono_use_reverb=DDSP_MONO_SYNTH_USE_REVERB,
                hex_use_reverb=DDSP_HEX_SYNTH_USE_REVERB,
                hex_reverb_length=REVERB_IR_LENGTH,
                target_audio=DDSP_HEX_SYNTH_TARGET_AUDIO,
                load_mono_synth_weights=HEX_SYNTH_LOAD_MONO_SYNTH_WEIGHTS,
                mono_synth_weight_paths=HEX_MONO_SYNTH_WEIGHT_PATHS):
        """
        Initialize DDSP hex synth module.

        Parameters
        ----------
        sr : float
            audio sampling rate
        hop_lenth : int
            samples to hop between frame starts
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
        eval_dict : dictionary
            contains flags for which evaluations to carry out
        mono_use_reverb : bool
            whether to use reverb on each mono synth
        hex_use_reverb : bool
            whether to use a final reverb on the summed output
        hex_reverb_length : int
            length of hex reverb impulse response in samples
        target_audio : str
            which audio to use as a ground truth (mix_audio or mic_audio)
        load_mono_synth_weights : bool
            if true, load mono synth weights (prompting to select)
        mono_synth_weight_paths : dict
            ['E'] : path to E string weights
            ...
            ['e'] : path to e string weights
        """
        super().__init__()      

        ####### All models should have the following defined ########

        self.name = "DDSP Hex Synth"

        # define keys to access inputs, outputs + labels (targets) of the network
        self.config = {'inputs' : ["E_f0", "E_loudness",
                                    "A_f0", "A_loudness",
                                    "D_f0", "D_loudness",
                                    "G_f0", "G_loudness",
                                    "B_f0", "B_loudness",
                                    "e_f0", "e_loudness"], 
                        'labels' : [target_audio], # compare against, mix audio
                        'outputs' : ["audio"], # model outputs synthesized audio               
        }

        # dictionary of evaluations to include in checkpoint
        self.eval_dict = eval_dict

        # set loss function
        self.loss_function = multi_scale_spectral_loss

        ##############################################################

        self.hex_use_reverb = hex_use_reverb

        self.string_synths = nn.ModuleDict()

        for string_let in GUITAR_STRING_LETTERS:
            synth = DDSPMonoSynth(sr=sr, 
                                hop_length=hop_length,
                                use_timbre=use_timbre, 
                                mlp_num_units=mlp_num_units, 
                                mlp_num_layers=mlp_num_layers, 
                                mlp_activation=mlp_activation,
                                gru_num_units=gru_num_units,
                                num_oscillators=num_oscillators,
                                num_filter_coeffs=num_filter_coeffs,
                                use_reverb=mono_use_reverb,
                                eval_dict=DDSP_MONO_SYNTH_EVAL_DICT)
            if load_mono_synth_weights:
                if mono_synth_weight_paths:
                    path = mono_synth_weight_paths[string_let]
                    synth.load_state_dict(save_load.load_weights_from_file(checkpoint_path=path))  
                else:  # if path dict is none, prompt user to select weights
                    print(f"Select weights for {string_let} string:")
                    synth.load_state_dict(save_load.load_weights_from_file())
                
            self.string_synths[string_let] = synth
        
        self.mono_output_key = "reverb_audio" if mono_use_reverb else "audio"
        if self.hex_use_reverb:
            self.config["outputs"] = ["reverb_audio"]
            self.reverb = TrainableReverb(reverb_length=hex_reverb_length)

        self.string_weighting = nn.Linear(6, 1, device=DEVICE) # how much to weight each string by when summing


    def forward(self, *inputs):
        """
        Process a dict of input data (f0s, loudnesses + timbre eventually) with 6 DDSP Mono Synths.

        Parameters
        ----------
        inputs: list of input tensor data 
            E_f0 (batch, frames)
            E_loudness (batch, frames)
            A_f0 (batch, frames)
            ...
            e_loudness(batch, frames)
        Returns
        ----------
        outputs : dict
            a['audio'] : summed audio (batch, samples)
            a['reverb_audio'] : reverberated summed audio (batch, samples)
        """

        # inputs is a list of the input args
        # construct a dictionary containing the input keys and their values
        # should this be resolved in model trainer somehow?
        input_keys = self.config['inputs']
        inputs_dict = { k:v for (k, v) in zip(input_keys, inputs)}

        output_string_audios = []
        for string_let in GUITAR_STRING_LETTERS:
            f0 = inputs_dict[f"{string_let}_f0"]
            loudness = inputs_dict[f"{string_let}_loudness"]
            string_output = self.string_synths[string_let](f0,loudness)
            output_string_audios.append(string_output[self.mono_output_key])

        audio_stack = torch.stack(output_string_audios, dim=-1) # shape: (batch, samples, 6)
        
        # weighted sum
        audio = self.string_weighting(audio_stack).squeeze(-1) # apply weighting and remove final dim of 1

        outputs = {}
        outputs['audio'] = audio
        if self.hex_use_reverb:
            reverb_audio = self.reverb(audio)
            outputs['reverb_audio'] = reverb_audio

        return outputs

