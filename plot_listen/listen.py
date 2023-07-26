# Author: Andy Wiggins <awiggins@drexel.edu>
# functions for sonifying

from IPython.display import Audio
import torch
from util import torch_to_numpy
from globals import SR

def play(arr, sr=SR):
	"""
	Returns Audio playback display object for given audio at sampling rate.

		Parameters
		----------
		arr : 1D numpy array
			array of audio samples
		sr : float
			sampling rate of arr

		Returns
		----------
		string_audio : IPython.display.Audio object
			Audio playback display
	"""
	# if it's a torch tensor, make numpy array
	if torch.is_tensor(arr):
		arr = torch_to_numpy(arr)
	
	return Audio(data=arr, rate=sr)