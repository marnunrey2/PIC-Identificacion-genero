import numpy as np


def mtRound(inp, precision=1):
    """
    mtRound
    Rounds the input to the specified precision.

    Parameters:
    - inp (numpy.ndarray or float): The input value or array to be rounded
    - precision (float, optional): The precision to round to. Default is 1.

    Returns:
    - out (numpy.ndarray or float): The rounded output

    Usage: out = mtRound(inp, precision)
    """
    return np.round(inp / precision) * precision
