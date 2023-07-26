# Author: Andy Wiggins <awiggins@drexel.edu>
# sweetcocoa implementation of DDSP filtered noise module



import numpy as np
import torch
import torch.nn as nn
import torch.fft


class SweetCocoaFilteredNoise(nn.Module):
    def __init__(self, frame_length = 64, attenuate_gain = 1e-2):
        super(SweetCocoaFilteredNoise, self).__init__()
        
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain
        
    def forward(self, z):
        """
        Compute linear-phase LTI-FVR (time-varient in terms of frame by frame) filter banks in batch from network output,
        and create time-varying filtered noise by overlap-add method.
        
        Argument:
            z['H'] : filter coefficient bank for each batch, which will be used for constructing linear-phase filter.
                - dimension : (batch_num, frame_num, filter_coeff_length)
        
        """
        
        batch_num, frame_num, filter_coeff_length = z['H'].shape
        device = z['H'].device
        self.filter_window = nn.Parameter(torch.hann_window(filter_coeff_length * 2 - 1, dtype = torch.float32), requires_grad = False).to(device)
        
        INPUT_FILTER_COEFFICIENT = z['H']
        
        # Desired linear-phase filter can be obtained by time-shifting a zero-phase form (especially to a causal form to be real-time),
        # which has zero imaginery part in the frequency response. 
        # Therefore, first we create a zero-phase filter in frequency domain.
        # Then, IDFT & make it causal form. length IDFT-ed signal size can be both even or odd, 
        # but we choose odd number such that a single sample can represent the center of impulse response.
        ZERO_PHASE_FR_BANK = INPUT_FILTER_COEFFICIENT.unsqueeze(-1).expand(batch_num, frame_num, filter_coeff_length, 2).contiguous() # (batch_num, frames, filter_len, complex_parts)
        ZERO_PHASE_FR_BANK[..., 1] = 0
        ZERO_PHASE_FR_BANK = torch.view_as_complex(ZERO_PHASE_FR_BANK.view(-1, filter_coeff_length, 2)) # complex tensor (batch * frames, filter len)
        ### AFW Removing dim=1 ###
        zero_phase_ir_bank = torch.fft.irfft(ZERO_PHASE_FR_BANK, n= filter_coeff_length * 2 - 1) # real tensor (batch * frames, filter len * 2 - 1)
        # zero_phase_ir_bank = torch.fft.irfft(ZERO_PHASE_FR_BANK, dim=1, n= filter_coeff_length * 2 - 1) # real tensor (batch * frames, filter len * 2 - 1)

        # Make linear phase causal impulse response & Hann-window it.
        # Then zero pad + DFT for linear convolution.
        linear_phase_ir_bank = zero_phase_ir_bank.roll(filter_coeff_length - 1, 1) # shape: (batch * frames, filter len * 2 - 1)
        windowed_linear_phase_ir_bank = linear_phase_ir_bank * self.filter_window.view(1, -1) # shape: (batch * frames, filter len * 2 - 1)
        zero_paded_windowed_linear_phase_ir_bank = nn.functional.pad(windowed_linear_phase_ir_bank, (0, self.frame_length - 1)) # shape: (batch * frames, filter len * 2 - 1 + hop_length = 192)
        ### AFW Removing dim=1 ###
        ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.view_as_real(torch.fft.rfft(zero_paded_windowed_linear_phase_ir_bank)) # shape: (batch * frames, 97, 2)
        # ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK = torch.view_as_real(torch.fft.rfft(zero_paded_windowed_linear_phase_ir_bank, dim=1)) # shape: (batch * frames, 97, 2)
        
        # Generate white noise & zero pad & DFT for linear convolution.
        noise = torch.rand(batch_num, frame_num, self.frame_length, dtype = torch.float32).view(-1, self.frame_length).to(device) * 2 - 1 # shape: (batch * frames, hop_length=64)
        zero_paded_noise = nn.functional.pad(noise, (0, filter_coeff_length * 2 - 2)) # (batch * frames, 64 + (65 * 2 - 2) = 192)
        ### AFW Removing dim=1 ###
        ZERO_PADED_NOISE = torch.view_as_real(torch.fft.rfft(zero_paded_noise)) # shape: (batch * frames, 97, 2)
        # ZERO_PADED_NOISE = torch.view_as_real(torch.fft.rfft(zero_paded_noise, dim=1)) # shape: (batch * frames, 97, 2)

        # Convolve & IDFT to make filtered noise frame, for each frame, noise band, and batch.
        FILTERED_NOISE = torch.zeros_like(ZERO_PADED_NOISE).to(device) # shape(batch * frames, 97, 2)
        FILTERED_NOISE[:, :, 0] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0] \
            - ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1]
        FILTERED_NOISE[:, :, 1] = ZERO_PADED_NOISE[:, :, 0] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 1] \
            + ZERO_PADED_NOISE[:, :, 1] * ZERO_PADED_WINDOWED_LINEAR_PHASE_FR_BANK[:, :, 0]
        
        # FILTERED_NOISE shape(batch * frames, 97, 2)
        FILTERED_NOISE = torch.view_as_complex(FILTERED_NOISE) # shape: complex (batch * frames, 97)
        ### AFW Removing dim=1 ###
        filtered_noise = torch.fft.irfft(FILTERED_NOISE).view(batch_num, frame_num, -1) * self.attenuate_gain  # shape = batch, frames, 2 * (97 - 1) = 192 )   
        # filtered_noise = torch.fft.irfft(FILTERED_NOISE, dim=1).view(batch_num, frame_num, -1) * self.attenuate_gain  # shape = batch, frames, 2 * (97 - 1) = 192 )   
                
        # Overlap-add to build time-varying filtered noise.
        overlap_add_filter = torch.eye(filtered_noise.shape[-1], requires_grad = False).unsqueeze(1).to(device)
        output_signal = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), 
                                                       overlap_add_filter, 
                                                       stride = self.frame_length, 
                                                       padding = 0).squeeze(1)
        
        
        output_signal = output_signal[:, :frame_num * self.frame_length] # crop output shape
        
        return output_signal