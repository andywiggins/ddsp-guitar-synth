# Author: Andy Wiggins <awiggins@drexel.edu>
# Function for multi-scale spectral loss

from torch import stft
from util import safe_log
from globals import *


def multi_scale_spectral_loss(x, y, 
                            num_losses=NUM_SPECTRAL_LOSSES, 
                            min_fft_size=MIN_SPECTRAL_LOSS_FFT_SIZE,
                            weighting=SPECTRAL_LOSS_ALPHA_WEIGHTING):
    """
    Computes multi-scale spectral loss between x and y

    Parameters
    ----------
    x : torch tensor (batch, samples)...can also be just (samples)
        tensor to compare loss function with (audio)
    y : torch tensor (batch, samples)...can also be just (samples)
        tensor to compare loss function with (audio)
    num_losses : int
        number of spectral losses to compute
    min_win_size : int
        starting window size, doubles each time
    weighting : float
        amount to weight log spectrogram comparison


    Returns
    ----------
    loss : tensor
        loss tensor
    """

    # initialize loss to zero
    loss = 0

    # get window sizes
    fft_sizes = [min_fft_size * (2**i) for i in range(num_losses)]

    # for each fft size
    for fft_size in fft_sizes:
        hop_length = int(fft_size / 4)  # overlap by 75%
        
        # compute stfts for x
        x_stft = stft(x, 
                    n_fft=fft_size, 
                    hop_length=hop_length, 
                    win_length=fft_size,
                    window=torch.hann_window(fft_size, device=x.device), #move window to gpu
                    normalized=True,
                    return_complex=True).abs()

        # compute stfts for y
        y_stft = stft(y, 
                    n_fft=fft_size, 
                    hop_length=hop_length, 
                    win_length=fft_size,
                    window=torch.hann_window(fft_size, device=y.device), #move window to gpu
                    normalized=True,
                    return_complex=True).abs()

        # compute linear and log losses, weight them
        linear_loss = (x_stft - y_stft).abs().mean() # taking the mean here means the loss is being scaled by batch size
        log_loss = (safe_log(x_stft) - safe_log(y_stft)).abs().mean() 
        L_i = linear_loss + weighting * log_loss

        # add on to total loss
        loss += L_i

    return loss
    
            




