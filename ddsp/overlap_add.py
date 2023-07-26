# Author: Andy Wiggins <awiggins@drexel.edu>
# Function for overlap adding

import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from util import crop_or_pad
from globals import *

def overlap_add(frame_arr, hop_length=HOP_LENGTH):
	"""
	Takes tensor of frames and overlap adds the frames. Adapted from here:
	https://github.com/sweetcocoa/ddsp-pytorch/blob/master/components/filtered_noise.py

	Parameters
	----------
	frame_arr : torch tensor (*, frames, frame_len) 
		last two dimensions will by overlap added to get to samples
	hop_length : int
	
	Returns
	----------
	sample_arr : tensor (*, samples)
		overlap add reuslt with last two dimensions reduced to samples
	"""
	frame_len = frame_arr.shape[-1]
	overlap_add_filter = torch.eye(frame_len, device=frame_arr.device)[:, None, :] # expanded identity matrix
	# look into how conv_transpose1d does what I want exactly
	
	sample_arr = F.conv_transpose1d(frame_arr.transpose(1, 2), 
									overlap_add_filter, 
									stride=hop_length).squeeze(1)
	
	return sample_arr