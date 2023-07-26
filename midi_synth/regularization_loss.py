# Author: Andy Wiggins <awiggins@drexel.edu>
# Function for multi-scale spectral loss

import torch
from globals import *


def regularization_loss(t, order=REGULARIZATION_LOSS_ORDER, weight=REGULARIZATION_LOSS_WEIGHT):
    """
    Computes regularization loss of tensor t

    Parameters
    ----------
    t : torch tensor (batch, *) 
        tensor to compute L1 reg loss of
    order : int
        1 for L1 loss
        2 for L2 loss
        etc.

    Returns
    ----------
    loss : tensor
        loss tensor
    """

    loss = weight * torch.norm(t, p=order)

    return loss
    
            




