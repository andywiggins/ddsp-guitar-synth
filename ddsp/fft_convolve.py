# Author: Andy Wiggins <awiggins@drexel.edu>
# Function for convolving two 1D tensors by multiplying their FFTs

import torch.nn as nn
import torch.fft
from globals import *
from util import crop_or_pad

def fft_convolve(x, h, output_len=None):
    """
    Takes two time domain signals (or batches of signals) and convolves them by multiplying their FFTs.

    Parameters
    ----------
    x : time domain signal tensor (*, samples) 
        Signal to convolve. will convolve along last dimension
    h : time domain signal tensor (*, samples) 
        other signal to convolve. will convolve along last dimension
    output_len : int
        length to pad or crop y to

    Returns
    ----------
    y : tensor (*, samples)
        time domain result of convolution in samples

    """        
    # pad lengths, get size for n_fft
    length = x.shape[-1] + h.shape[-1] - 1
    n_fft = 2 ** (length - 1).bit_length() # get next highest power of 2

    # compute fft of each signal
    X = torch.fft.fft(x, n=n_fft) # automatically pads when we give n
    H = torch.fft.fft(h, n=n_fft)

    # multiply in freq domain
    Y = X * H

    # inverse fft, take real part
    y = torch.fft.ifft(Y).real

    # change output length of y
    if output_len is not None:
        y = crop_or_pad(y, output_len)

    return y





