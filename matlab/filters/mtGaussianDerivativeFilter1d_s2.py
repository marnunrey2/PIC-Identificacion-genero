from .mtGaussianDerivativeFilter1d_s0 import mtGaussianDerivativeFilter1d_s0


def mtGaussianDerivativeFilter1d_s2(x, sigma):
    """
    mtGaussianDerivativeFilter1d_s2
    Generate discrete second-order 1D Gaussian Derivative filter

    Parameters:
    - x (1D array): array of x values
    - sigma (float): Standard deviation of Gaussian used to generate filter

    Returns:
    - s2 (1D array): second order 1D filter

    Usage: s2 = mtGaussianDerivativeFilter1d_s2(x, sigma)
    """
    # Compute the zeroth order 1D Gaussian Derivative filter
    s0 = mtGaussianDerivativeFilter1d_s0(x, sigma)

    # Compute the second order 1D Gaussian Derivative filter
    s2 = ((x**2 - sigma**2) / (sigma**4)) * s0

    return s2
