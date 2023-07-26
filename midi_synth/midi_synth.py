# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch synth module with midi-like conditioning

from midi_synth.mono_network import MonoNetwork
from midi_synth.mono_network_small import MonoNetworkSmall
from ddsp.harmonic_oscillator import HarmonicOscillator
from ddsp.filtered_noise import FilteredNoise
from ddsp.trainable_reverb import TrainableReverb
from ddsp.multi_scale_spectral_loss import multi_scale_spectral_loss
from midi_synth.midi_util import midi_to_hz
from midi_synth.context_network import ContextNetwork
from midi_synth.regularization_loss import regularization_loss
import torch.nn as nn
import torchaudio.transforms
import save_load
from globals import *

class MidiSynth(nn.Module):
    """
    Torch nn module for a ddsp-style synth that takes in conditioning in a midi-like format.
    """
    def __init__(self,
                sr=SR,
                hop_length=HOP_LENGTH,
                mlp_num_units=MLP_NUM_UNITS, 
                mlp_num_layers=MLP_NUM_LAYERS, 
                mlp_activation=MLP_ACTIVATION,
                gru_num_units=GRU_NUM_UNITS,
                use_small_mono_net=USE_SMALL_MONO_NET,
                sm_linear_1_size=MONO_NET_SM_LINEAR_1_SIZE,
                sm_linear_2_size=MONO_NET_SM_LINEAR_2_SIZE,
                sm_gru_num_units=MONO_NET_SM_GRU_NUM_UNITS,
                use_string_embedding=MIDI_SYNTH_USE_STRING_EMBEDDING,
                use_mfcc_input=MIDI_SYNTH_USE_MFCC_INPUT,
                string_embedding_num=STRING_EMBEDDING_NUM,
                string_embedding_size=STRING_EMBEDDING_SIZE,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS,
                use_context_net=USE_CONTEXT_NET,
                reverb_length=REVERB_IR_LENGTH,
                learn_pitch_adjustment=MIDI_SYNTH_LEARN_PITCH_ADJUSTMENT,
                eval_dict=MIDI_SYNTH_EVAL_DICT,
                target_audio=MIDI_SYNTH_TARGET_AUDIO,
                parallelize_over_strings=PARALLELIZE_OVER_STRINGS):
        """
        Initialize midi synth module.

        Parameters
        ----------
        sr : float
            audio sampling rate
        hop_lenth : int
            samples to hop between frame starts
        mlp_num_units : int
            number of units in each mlp layer
        mlp_num_layers : int
            number of layers in each mlp
        mlp_activation : torch.nn module
            activation to apply in mlp layers
        gru_num_units : int
            number of units in the gru's hidden layer
        use_small_mono_net : bool
            if True, use the smaller mono net based on piano ddsp paper
        sm_linear_1_size : int
            size of 1st dense layer, if using the small mono net
        sm_linear_2_size : int
            size of 2nd dense layer, if using the small mono net
        sm_gru_num_units : int
            unit size of gru, if using the small mono net
        use_string_embedding : bool
            if True, use string index as input to mono network
        use_mfcc_input : bool
            if true, then synth expects audio as input so that mfccs can be calculated
        string_embedding_num : int
            number of possible strings (6 for now, but should it be 6 x num_guitars eventually?)
        string_embedding_size : int
            size of the string embedding vector
        num_oscillators : int
            number of oscillators to get harm distribution for (corresponds to number of oscillators in synth)
        filter_coeffs : int
            number of filter coefficients to output for use in filtered noise synth 
        reverb_length : int
            length of reverb impulse response in samples
        learn_pitch_adjustment : bool
            if True, the pitch conditioning can be adjusted by a network
        eval_dict : dictionary
            contains flags for which evaluations to carry out
        target_audio : str
            which audio to use as a ground truth (mix_audio or mic_audio
        parallelize_over_strings : bool
            if true, combine batch and string dims before passing to mono net)
        """
        super().__init__()      

        ####### All models should have the following defined ########

        self.name = "MIDI Synth"

        # define keys to access inputs, outputs + labels (targets) of the network
        self.config = {'inputs' : ["conditioning"], 
                        'labels' : [target_audio], # compare against, mix/mic audio
                        'outputs' : ["audio"], # model outputs synthesized audio               
        }

        # dictionary of evaluations to include in checkpoint
        self.eval_dict = eval_dict

        # set loss function
        self.loss_function = multi_scale_spectral_loss

        ##############################################################

        self.sr = sr
        self.hop_length = hop_length

        self.parallelize_over_strings = parallelize_over_strings
        self.use_string_embedding = use_string_embedding

        self.use_context_net = use_context_net
        if self.use_context_net:
            self.context_network = ContextNetwork()

        self.use_mfcc_input = use_mfcc_input
        if self.use_mfcc_input:
            print("Using mfcc as additional input to synth.")
            self.config['inputs'] += [target_audio] # add "mic_audio" to inputs, for instance, so that mfcc can be calculated

            self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=self.sr,
                                            n_mfcc=N_MFCC,
                                            melkwargs={"n_fft": N_FFT, 
                                                    "hop_length": self.hop_length, 
                                                    "n_mels": N_MELS, 
                                                    "center": True}
            )

        if use_small_mono_net:
            self.mono_network = MonoNetworkSmall(
                                linear_1_size=sm_linear_1_size,
                                linear_2_size=sm_linear_1_size,
                                gru_num_units=sm_gru_num_units,
                                use_string_embedding=use_string_embedding,
                                use_context_net=use_context_net,
                                string_embedding_num=string_embedding_num,
                                string_embedding_size=string_embedding_size,
                                num_oscillators=num_oscillators,
                                num_filter_coeffs=num_filter_coeffs,
                                use_mfcc_input=use_mfcc_input)
        else:
            self.mono_network = MonoNetwork(
                                    mlp_num_units=mlp_num_units, 
                                    mlp_num_layers=mlp_num_layers, 
                                    mlp_activation=mlp_activation,
                                    gru_num_units=gru_num_units,
                                    use_string_embedding=use_string_embedding,
                                    string_embedding_num=string_embedding_num,
                                    string_embedding_size=string_embedding_size,
                                    num_oscillators=num_oscillators,
                                    num_filter_coeffs=num_filter_coeffs,
                                    use_context_net=use_context_net,
                                    use_mfcc_input=use_mfcc_input,
                                    learn_pitch_adjustment=learn_pitch_adjustment)

        ### test with sweetcocoa code to see if there's a glitch in my ddsp components
        self.harmonic_oscillator = HarmonicOscillator(sr=sr, hop_length=hop_length, inharmonicity=INHARMONICITY)

        self.filtered_noise = FilteredNoise(hop_length=hop_length)
        
        self.reverb = TrainableReverb(reverb_length=reverb_length)

    def forward(self, conditioning, input_audio=None):
        """
        Process a conditioning tensor (and optionally audio) with 6 DDSP Mono Synths.

        Parameters
        ----------
        conditioning: tensor (batch, frames, num_strings=6, pitch/vel=2)
            pitch and onset velocity labels across 6 strings

        Returns
        ----------
        outputs : dict
            a['audio'] : output audio (batch, samples)
        """
        # normalize conditioning
        conditioning = conditioning / MIDI_NORM

        # ### first, parallelize guitar strings along with batch
        if self.parallelize_over_strings:
            return self.forward_parallel(conditioning, input_audio=input_audio)
        
        # get context
        if self.use_context_net:
            context = self.context_network(conditioning)
        else:
            context = None

        # calculated mfcc from input audio
        if self.use_mfcc_input:
            mfcc = self.mfcc_transform(input_audio)[:,:,:-1]  # I'm getting one extra frame, so slice to (batch, n_mfcc, frames)
            mfcc = torch.transpose(mfcc, 1, 2) # (batch, frames, n_mfcc) # put n_mfcc last since the linear layers assume channel last
        else:
            mfcc = None
        
        # get audio for each guitar string
        string_audios = []
        for i in range(6):

            # extract string conditioning and index label
            string_cond = conditioning[:, :, i, :] # (batch, frames, 2)

            if self.use_string_embedding:
                # expand idx to batch size
                string_idx = torch.IntTensor([i]).expand(string_cond.shape[0]).to(conditioning.device) # (batch)
            else:
                string_idx = None

            params = self.mono_network(string_cond, string_idx=string_idx, mfcc=mfcc, context=context)       

            harmonic = self.harmonic_oscillator(params, string_idx=string_idx) # (batch, samples)

            noise = self.filtered_noise(params) # (batch, samples)

            string_audio = harmonic + noise # (batch, samples)

            string_audios.append(string_audio)


        # sum string audios
        string_audios = torch.stack(string_audios, dim=-1) # (batch, samples, 6)
        dry_audio = torch.sum(string_audios, dim=-1) # (batch, samples)

        audio = self.reverb(dry_audio)

        outputs = {}
        outputs['audio'] = audio

        return outputs
    
    def forward_parallel(self, conditioning, input_audio=None):
        """
        The forward method but parallelized over the 6 guitar strings (as well as batches)
        Process a conditioning tensor (and optionally audio) with 6 DDSP Mono Synths.

        Parameters
        ----------
        conditioning: tensor (batch, frames, num_strings=6, pitch/vel=2)
            pitch and onset velocity labels across 6 strings
            assumed to be normalized in range [0,1]

        Returns
        ----------
        outputs : dict
            a['audio'] : output audio (batch, samples)
        """

        # get context
        if self.use_context_net:
            context = self.context_network(conditioning) # (batch, frames, c=32)
            context = context.unsqueeze(0).expand(6, -1, -1, -1).flatten(0,1) # expand batch dim to size=(batch*6)
        else:
            context = None

        batch = conditioning.shape[0]

        # first, parallelize guitar strings along with batch
        
        # make conditioning have shape: (batch x 6, frames, 2)
        # note: permute order makes it so that after flattening we have all string_idx=0, then all string_idx=1 etc.
        conditioning = conditioning.permute(2, 0, 1, 3).flatten(0, 1) # (batch * 6, frames, 2)

        # calculate mfcc from input audio
        if self.use_mfcc_input:
            mfcc = self.mfcc_transform(input_audio)[:,:,:-1]  # I'm getting one extra frame, so slice to (batch, n_mfcc, frames)
            mfcc = torch.transpose(mfcc, 1, 2) # (batch, frames, n_mfcc) # put n_mfcc last since the linear layers assume channel last
            mfcc = mfcc.unsqueeze(0).expand(6, -1, -1, -1).flatten(0,1) # expand batch dim to size=(batch*6)
        else:
            mfcc = None

        if self.use_string_embedding:
            string_idxs = []
            for i in range(6):
                string_idxs.append(torch.IntTensor([i]).expand(batch).to(conditioning.device)) # (batch)

            string_idx = torch.stack(string_idxs, dim=0) # (6, batch)
            string_idx = string_idx.flatten().to(conditioning.device) # (batch * 6)
            self.string_idx = string_idx
        else:
            string_idx = None

        params = self.mono_network(conditioning, string_idx=string_idx, mfcc=mfcc, context=context)       

        harmonic = self.harmonic_oscillator(params, string_idx=string_idx) # (batch * 6, samples)

        noise = self.filtered_noise(params) # (batch * 6, samples)

        string_audio = harmonic + noise # (batch * 6, samples)

        # revert shape back to (6, batch, samples)
        string_audio = torch.unflatten(string_audio, 0, (6, batch))

        # sum string audios
        dry_audio = torch.sum(string_audio, dim=0) # (batch, samples)

        audio = self.reverb(dry_audio)

        outputs = {}
        outputs['audio'] = audio

        # optionally, save off predicted params (per string)
        if MIDI_SYNTH_OUTPUT_PREDICTED_PARAMS:

            # overall amplitude
            outputs['overall_amplitude'] = torch.unflatten(params['overall_amplitude'], 0, (6, batch))
            # harmonic distribution
            outputs['harm_distr'] = torch.unflatten(params['harm_distr'], 0, (6, batch))
            # H
            outputs['H'] = torch.unflatten(params['H'], 0, (6, batch))
            # f0
            outputs['f0'] = torch.unflatten(params['f0'], 0, (6, batch))

        if MIDI_SYNTH_OUTPUT_FACTORED_AUDIO:
            # string audio
            outputs['string_audio'] = string_audio # already been unflattened above
            # harmonic
            outputs['harmonic'] = torch.unflatten(harmonic, 0, (6, batch))
            # noise
            outputs['noise'] = torch.unflatten(noise, 0, (6, batch))




        return outputs
    
    def compute_regularization_loss(self):
        """
        Returns regularization loss for any model parameters that need to be regularized.

        For now, just the reverb fir. 

        Returns
        ----------
        loss : tensor
            the calculated regularization loss
        """

        return regularization_loss(self.reverb.generate_fir())

