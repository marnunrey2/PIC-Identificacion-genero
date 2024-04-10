from .mtGaussianDerivativeFilter1d_s0 import mtGaussianDerivativeFilter1d_s0


def mtGaussianDerivativeFilter1d_s1(x, sigma):
    """
    mtGaussianDerivativeFilter1d_s1
    Generate discrete first-order 1D Gaussian Derivative filter

    Parameters:
    - x (1D array): array of x values
    - sigma (float): Standard deviation of Gaussian used to generate filter

    Returns:
    - s1 (1D array): first order 1D filter

    Usage: s1 = mtGaussianDerivativeFilter1d_s1(x, sigma)
    """
    # Compute the zeroth order 1D Gaussian Derivative filter
    s0 = mtGaussianDerivativeFilter1d_s0(x, sigma)

    # Compute the first order 1D Gaussian Derivative filter
    s1 = (x / (sigma**2)) * s0

    return s1
