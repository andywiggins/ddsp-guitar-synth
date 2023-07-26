# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn module for filtered noise
import torch.nn as nn
import torch.fft
from util import crop_or_pad
from ddsp.fft_convolve import fft_convolve
from ddsp.overlap_add import overlap_add
from globals import *

class FilteredNoise(nn.Module):
    """
    Torch nn module for filtered noise as described in DDSP paper.
    """
    def __init__(self, hop_length=HOP_LENGTH):
        """
        Initialize filtered noise module.

        Parameters
        ----------
        hop_length : int
            samples to hop from frame to frame
        """
        super().__init__()

        self.hop_length=HOP_LENGTH


    def forward(self, p, impulse_length=None):
        """
        Pass input through filtered noise. Applied a time varying filter to a noise signal.

        Parameters
        ----------
        p : dict
		    p['H'] : filter coefficients (batch, frames, filter_len)
        impulse_length : None
            length of impulse to convolve with
            if None, will default to 2 * filter_coeff - 1
            should probalby be >= default

        Returns
        ----------
        filtered_noise : tensor (batch, samples)
        	Output audio of noise filtered with filter_coefficients
        """

        H = p['H']
        filter_coefficent_len = H.shape[-1]

        # Note -- fft of a real, even signal is even and real:
        #   we stack zeros for imaginary part (real)
        #   take inverse using rfft which assumes we gave half of an even signal 

        H = torch.stack([H, torch.zeros_like(H)], -1)
        H = torch.view_as_complex(H) 
        h = torch.fft.irfft(H, n=filter_coefficent_len * 2 - 1) # (batch, frame, filter_len)
        filter_len = h.shape[-1]

        # roll impulse response to start at time t=0
        h = torch.roll(h, filter_len// 2, -1)
        # window
        h *= torch.hann_window(filter_len, device=h.device)
        # crop or pad impulse
        if impulse_length is not None:
            h = crop_or_pad(h, impulse_length)
        # roll it back
        h = torch.roll(h, -filter_len//2, -1)

        # create noise tensor
        noise = torch.rand(h.shape[0], h.shape[1], self.hop_length, device=h.device) # (batch, frames, hop_length)
        noise = noise * 2 - 1 # scale from [0,1] to [-1, -1]

        # filter via convolution in freq domain
        filtered_noise_arr = fft_convolve(noise, h, output_len=self.hop_length) # (batch, frames, new_len >= hop_length)

        # overlap-add the filtered_noise arr into audio
        # note: since I am cropping each frame length I don't really need to "overlap add"
        # I could probably just do this by reshaping.
        filtered_noise = overlap_add(filtered_noise_arr, hop_length=self.hop_length)

        return filtered_noise























    

