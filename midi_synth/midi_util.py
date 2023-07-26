# Author: Andy Wiggins <awiggins@drexel.edu>
# midi utility functions

def midi_to_hz(m):
    """
    Takes in a scalar, numpy array or torch tensor m of midi pitch values and converts it to frequencies in Hertz.

    Parameters
    ----------
    x : number/array/tensor
        array to be converted

    Returns
    ----------
    y : array/tensor
        y is in Hertz 
    """
    return 2 ** ((m - 69) / 12) * 440