# Author: Andy Wiggins <awiggins@drexel.edu>
# utility functions

import numpy as np
import torch
import torch.nn.functional as F
import os
import shutil
from globals import *

def crop_or_pad(x, size):
    """
    Takes in a numpy array or torch tensor x and slices pads it along dim=-1 to have length size.

    Parameters
    ----------
    x : numpy array
        array to be cropped/padded
    size : int
        desired length for x

    Returns
    ----------
    updated_x : numpy array
        x, sliced/padded to desired size 
    """
    # if x is a list, cast to numpy array
    if type(x) == list:
        x = np.array(x)

    # set library to torch or numpy
    lib = F if torch.is_tensor(x) else np  # use torch or numpy to do the padding

    # get size
    x_size = x.shape[-1]

    # crop or pad
    if x_size > size:
        updated_x = x[..., :size]
    elif x_size < size:
        padding_len = size - x_size
        # padding works differently in pytorch and numpy
        if lib == np:
            padding = tuple([(0,0) for dim in x.shape[:-1]] + [(0,padding_len)])
        else:
            padding = (0,padding_len)
        updated_x = lib.pad(x, padding, "constant")

    else:
        updated_x = x  
    return updated_x

def chunk_arr(x, chunk_size, omit_incomplete_chunks=OMIT_INCOMPLETE_CHUNKS):
    """
    Takes in a 1d array numpy x and splits it into chunks of of a given size. Returns a 2d array where the first dimension is # of chunks. Pads the final chunk with zeros if necessary.

    Parameters
    ----------
    x : numpy array to chunk
        array to be chunked
    chunk_size : int
        desired length for x.
    omit_incomplete_chunks : bool
        if true, use the floor to only have only complete chunks 

    Returns
    ----------
    chunked_arr : numpy array
        Shape = (num_chunks, chunk_size) 
    """
    if omit_incomplete_chunks:
        num_chunks = int(np.floor(len(x) / chunk_size))
        required_len = int(num_chunks * chunk_size) # crop to this length
        x_len_adjusted = x[:required_len]
    else:
        num_chunks = int(np.ceil(len(x) / chunk_size))
        required_len = num_chunks * chunk_size # pad to this length
        padding_len = int(round(required_len - len(x)))
        x_len_adjusted = np.pad(x, (0, padding_len), "constant")
    chunked_arr = np.array(np.split(x_len_adjusted, num_chunks))
    return chunked_arr

def torch_to_numpy(x):
    """
    Takes in a torch tensor x and converts to numpy array.

    Parameters
    ----------
    x : torch tensor
        tensor to be converted

    Returns
    ----------
    x_np : numpy array
        converted array
    """
    return x.cpu().detach().numpy()

def numpy_to_torch(x, device=DEVICE):
    """
    Takes in a numpy array x and converts to torch tensor. If passed in a torch tesnor already, returns it on device

    Parameters
    ----------
    x : np array
        array to be converted
    device: device
        desired device, default is DEVICE in global


    Returns
    ----------
    x_t : torch tensor
        converted tensor
    """
    return torch.from_numpy(x).to(device)

def safe_log(x, eps=EPS):
    """
    Takes log without resulting in a divide by zero error.

    Parameters
    ----------
    x : torch tensor
        thing to take log of

    Returns
    ----------
    log_x : torch tensor
        log of x
    """
    return torch.log(x + eps)

def print_globals():
    """
    Prints all the global variables, excluding built-in python things.
    """
    g = globals()
    for key in g:
        if key[0].isupper():
            if key[0] != "_" and key != "g" and key != "key" and key != "In" and key != "Out" and str(g[key])[0] != "<":
                print(key, "=", g[key])
            elif key[0] != "_" and key != "g" and key != "key" and key != "In" and key != "Out" and str(g[key])[0] == "<" and str(g[key])[-1] == ">":
                print(key, "=", simple_name_of_obj(g[key]))

def simple_name_of_obj(obj):
    """
    If an object's __str__ call results in something like <function multi_scale_spectral_loss at 0x7f961cfe3d40>, print just the name multi_scale_spectral_loss 

    Parameters
    ----------
    obj : any object
        object to be printed

    Returns
    ----------
    s : string
        string version of object
    """
    s = str(obj)
    if s[0] == "<" and s[-1] == ">":
        s = s.split(" ")[1].split(">")[0] # we want the string in between the first two spaces
    return s

def create_dir(path):
    """
    Creates directory (and path to it) if it does not exist

    Parameters
    ----------
    path : string
        path containing directories that need to be created 
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
def delete_dir(path):
    """
    Delete directory and contents recursively

    Parameters
    ----------
    path : string
        path containing directories + contents to be deleted
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def count_params(model):
    """
    Takes in a torch nn model and and returns a dict of the param count for each submodule

    Parameters
    ----------
    model : torch nn model
        model to count params in

    Returns
    ----------
    total : int
        total number of params
    counts : dict
        a parameter count dictionary containing
        counts[submodule_name] = num_params 
    """
    counts = {}
    total = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            module_name = name.split(".")[0]
            param_size = len(param.flatten())
            total += param_size
            if not module_name in counts:
                counts[module_name] = param_size
            else:
                counts[module_name] += param_size
        
    return total, counts

            


    


    






