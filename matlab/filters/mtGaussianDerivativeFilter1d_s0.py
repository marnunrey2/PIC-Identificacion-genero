import numpy as np


def mtGaussianDerivativeFilter1d_s0(x, sigma):
    """
    mtGaussianDerivativeFilter1d_s0
    Generate discrete zeroth-order 1D Gaussian Derivative filter

    Parameters:
    - x (1D array): array of x values
    - sigma (float): Standard deviation of Gaussian used to generate filter

    Returns:
    - s0 (1D array): zeroth order 1D filter

    Usage: s0 = mtGaussianDerivativeFilter1d_s0(x, sigma)
    """
    Cs = 1 / (np.sqrt(2 * np.pi) * sigma)
    s0 = Cs * np.exp(-(x**2) / (2 * sigma**2))
    return s0
