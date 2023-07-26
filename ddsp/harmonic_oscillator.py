# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn module class for harmonic oscillator module

import torch.nn as nn
import torch
from globals import *

class HarmonicOscillator(nn.Module):
    """
    Torch nn module for harmonic oscillator as described in DDSP paper.
    """
    def __init__(self, sr=SR, hop_length=HOP_LENGTH, inharmonicity=INHARMONICITY):
        """
        Initialize harmonic oscillator module.

        Parameters
        ----------
        sr : float
            sampling rate
        hop_length : int
            samples to hop from frame to frame
        inharmonicity : str or None
            None to not use inharmonicity
            "beta" to use 1 param learnable param beta
        """
        super().__init__()

        self.sr = sr
        self.hop_length = hop_length

        self.upsample = nn.Upsample(scale_factor=hop_length, mode="linear", align_corners=False)

        self.freeze_inharm_params = FREEZE_INHARM_PARAMS


        # in harmonicity settings
        self.inharmonicity = inharmonicity

        if self.inharmonicity == "global":
		    # value B in:     f_n = n * f_0(1 + B * n^2)^(0.5)
            # see https://www.uwec.edu/files/7224/Musical-string-inharmonicity-Chris-Murray.pdf
            self.inharm_beta = nn.Parameter(
            torch.logit(torch.tensor([INHARM_BETA_INIT])), # logit to get pre sigmoid value
            requires_grad= (not self.freeze_inharm_params),
        ) 
            
        if self.inharmonicity == "per-string":

		    # value B in:   f_n = n * f_0(1 + B * n^2)^(0.5)
            # use logit to get pre sigmoid values
            self.inharm_betas = nn.ParameterList([nn.Parameter(torch.logit(torch.tensor(b)), requires_grad=(not self.freeze_inharm_params)) for b in PER_STRING_INHARM_BETA_INIT])
            


    def forward(self, p, string_idx=None):
        """
        Pass input through harmonic oscillator.

        Parameters
        ----------
        p : dict
		    p['f0'] : f0 envelope tensor (batch, frames)
            p['overall_amplitude'] : overall amplitude (batch, frames)
            p['harm_distr'] : harmonic distribution (sums to 1) tensor (batch, # harmonics, frames)

        string_idx : Int tensor (batch)
            the string index of each batch item

        Returns
        ----------
        audio : tensor (batch, samples)
        	Output audio from harmonic oscillator

        """
        f0 = p['f0']
        overall_amplitude = p['overall_amplitude']
        harm_distr = p['harm_distr']

        # get batch size
        batch_size = f0.shape[0]
        
        # get number of oscillators from shape of harmonic distribution
        num_osc = harm_distr.shape[1]

        # compute harmonic multipliers, using inharmonicity if desired
        harmonic_mults = torch.arange(1, num_osc + 1, dtype=DEFAULT_TORCH_DTYPE, device=f0.device) # numbers of the harmonics, shape = (num_osc)

        # adjust harmonic mults with 1 global inharm beta
        if self.inharmonicity == "global": # replace harmonic_nums with values with inharmonicity factored in
            beta = torch.sigmoid(self.inharm_beta) # squash param to get beta between 0 and 1
            harmonic_mults *= (1 + beta * harmonic_mults**2) ** 0.5
            harmonic_mults = harmonic_mults[None, :, None] # shape: (batches=1, num_osc, frames=1)
        
        # adjust harmonic mults with 6 string-specific betas
        elif self.inharmonicity == "per-string":
            if string_idx == None:
                print("String index is required oscillator input for per-string inharmonicity.")
                exit(0)
            beta_list = []
            for idx in string_idx: # loop over batch (not ideal)
                beta_list.append(torch.sigmoid(self.inharm_betas[idx]))
            betas = torch.stack(beta_list, dim=0) # shape: (batch)
            harmonic_mults = harmonic_mults[None, :, None] # shape: (batches=1, num_osc, frames=1)
            betas = betas[:, None, None] # shape: (batches, num_osc=1, frames=1)
            harmonic_mults = harmonic_mults * ((1 + betas * harmonic_mults**2) ** 0.5)

        # create tensor of freq values for each harmonic   
        freqs = f0[:, None, :] *  harmonic_mults # shape = (batches, num_osc, frames)

        # upsample freqs to audio rate
        # note: upsample treats num_osc as channel dimension (ignored) which is fine
        audio_rate_freqs = self.upsample(freqs) # shape = (batches, num_osc, samples)

        # compute instananeous phase from audio rate freqs
        # we sum over samples
        inst_phases = torch.cumsum(audio_rate_freqs / self.sr, dim=2) # shape = (batches, num_osc, samples)
        
        # get amplitudes by scaling harm_distr by overall amplitude and then antialiasing
        amplitudes = overall_amplitude[:, None, :] * harm_distr  #  shape: (batch, num_oscillators, frames)
        antialias_mask = (freqs < (self.sr / 2)) # bool: (batch, num_osc, frames)
        amplitudes *= antialias_mask # (batches, num_osc, frames)

        # upsample harmonic distribution envelopes
        # note: upsample treats num_osc as channel dimension (ignored) which is fine
        audio_rate_amplitudes = self.upsample(amplitudes)

        # build harmonic sinusoid bank
        sinusoids = torch.sin(2 * torch.pi * inst_phases) * audio_rate_amplitudes # (batch, num_osc, samples)

        # sum sinusoids
        audio = torch.sum(sinusoids, dim=1) # (batch, samples)

        return audio



    

