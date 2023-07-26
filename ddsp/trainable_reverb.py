# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn module class for trainable reverb module
# Based on the implementation from https://github.com/sweetcocoa/ddsp-pytorch/blob/master/components/reverb.py

import torch.nn as nn
import torch.fft
from ddsp.fft_convolve import fft_convolve
from globals import *

class TrainableReverb(nn.Module):
    """
    Torch nn module for trainable reverb as described in DDSP paper.

    Based on implementation from: https://github.com/sweetcocoa/ddsp-pytorch/blob/master/components/reverb.py
    """
    def __init__(self, reverb_length=REVERB_IR_LENGTH, device=DEVICE):
        """
        Initialize trainable reverb module.

        Parameters
        ----------
        reverb_length : int
            length of reverb impulse response, in samples
        device : torch device
            CPU or GPU
        """

        super(TrainableReverb, self).__init__()

        # default reverb length is set to 3sec.
        # thus this model can max out t60 to 3sec, which corresponds to rich chamber characters.
        self.reverb_length = reverb_length
        self.device = device

        # impulse response of reverb.
        self.fir = nn.Parameter(
            torch.rand(1, self.reverb_length, dtype=torch.float32).to(self.device) * 2 - 1,
            requires_grad=True,
        )

        # Initialized drywet to around 26%.
        # but equal-loudness crossfade between identity impulse and fir reverb impulse is not implemented yet.
        self.drywet = nn.Parameter(
            torch.tensor([-1.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

        # Initialized decay to 5, to make t60 = 1sec.
        self.decay = nn.Parameter(
            torch.tensor([3.0], dtype=torch.float32).to(self.device), requires_grad=True
        )

    def forward(self, input_signal):
        """
        Pass input through trainable reverb.

        Parameters
        ----------
        input_signal : tensor (batch, samples)
		    audio to reverberate

        Returns
        ----------
        output_signal : tensor (batch, samples)
        	reverb audio
        """

        # Send batch of input signals in time domain to frequency domain.
        # Appropriate zero padding is required for linear convolution.
        zero_pad_input_signal = nn.functional.pad(input_signal, (0, self.fir.shape[-1] - 1))
        INPUT_SIGNAL = torch.view_as_real(torch.fft.rfft(zero_pad_input_signal))

        # compute final fir based on params
        final_fir = self.generate_fir()

        zero_pad_final_fir = nn.functional.pad(final_fir, (0, input_signal.shape[-1] - 1))

        FIR = torch.view_as_real(torch.fft.rfft(zero_pad_final_fir))

        # Convolve and inverse FFT to get original signal.
        OUTPUT_SIGNAL = torch.zeros_like(INPUT_SIGNAL).to(self.device)
        OUTPUT_SIGNAL[:, :, 0] = (
            INPUT_SIGNAL[:, :, 0] * FIR[:, :, 0] - INPUT_SIGNAL[:, :, 1] * FIR[:, :, 1]
        )
        OUTPUT_SIGNAL[:, :, 1] = (
            INPUT_SIGNAL[:, :, 0] * FIR[:, :, 1] + INPUT_SIGNAL[:, :, 1] * FIR[:, :, 0]
        )

        output_signal = torch.fft.irfft(torch.view_as_complex(OUTPUT_SIGNAL))

        output_signal = output_signal[:,:input_signal.shape[-1]] # crop to number of samples from input

        return output_signal
    
    def generate_fir(self):
        """
        Compute the final fir based on the reverb params.

        Returns
        ----------
        final_fir : tensor (batch, samples)
        	reverb audio
        """

        # Build decaying impulse response and send it to frequency domain.
        # Dry-wet mixing is done by mixing impulse response, rather than mixing at the final stage.

        decay_envelope = torch.exp(
            -(torch.exp(self.decay) + 2)
            * torch.linspace(0, 1, self.reverb_length, dtype=torch.float32).to(self.device)
        )
        decay_fir = self.fir * decay_envelope

        ir_identity = torch.zeros(1, decay_fir.shape[-1]).to(self.device)
        ir_identity[:, 0] = 1

        # Equal-loudness(intensity) crossfade between to ir.
        final_fir = (
            torch.sigmoid(self.drywet) * decay_fir + (1 - torch.sigmoid(self.drywet)) * ir_identity
        )

        return final_fir

        
    

        





    

