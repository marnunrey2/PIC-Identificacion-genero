import numpy as np
from .mtGaussianDerivativeFilter1d_s0 import mtGaussianDerivativeFilter1d_s0
from .mtGaussianDerivativeFilter1d_s1 import mtGaussianDerivativeFilter1d_s1
from .mtGaussianDerivativeFilter1d_s2 import mtGaussianDerivativeFilter1d_s2
from .mtDownsampleFilter import mtDownsampleFilter


def mtGaussianDerivativeFilters1d(sigma):
    """
    mtGaussianDerivativeFilters1d
    Generates discrete zeroth, first, and second order 1D Gaussian Derivative filters

    Parameters:
    - sigma (float): Standard deviation of Gaussian used to generate filters

    Returns:
    - s0 (1D array): zeroth order 1D filter
    - s1 (1D array): first-order 1D filter
    - s2 (1D array): second-order 1D filter

    Usage: s0, s1, s2 = mtGaussianDerivativeFilters1d(sigma)
    """

    # Internal parameters
    # scaleFactor: Number of bins in higher resolution filter for each bin in
    # original resolution filter
    # widthMulti: Filters are centered on the middle bin. The extent of the filter
    # either side of this central pixel is widthMulti standard deviations
    scaleFactor = 10
    widthMulti = 5
    halfWidth = (np.ceil(sigma) * widthMulti) + 1

    # Set x values for each bin of high resolution filter. Note that the x limits
    # are the same, we just generate more bins between them
    xHighRes = np.arange(-halfWidth, halfWidth + 1, 1 / scaleFactor)

    # Zeroth order filter
    s0HighRes = mtGaussianDerivativeFilter1d_s0(xHighRes, sigma)
    s0 = mtDownsampleFilter(s0HighRes, scaleFactor)
    # First-order filter
    s1 = None
    if True:  # Assuming always compute first-order filter
        s1R = mtGaussianDerivativeFilter1d_s1(xHighRes, sigma)
        s1 = mtDownsampleFilter(s1R, scaleFactor)
    # Second-order filter
    s2 = None
    if True:  # Assuming always compute second-order filter
        s2R = mtGaussianDerivativeFilter1d_s2(xHighRes, sigma)
        s2 = mtDownsampleFilter(s2R, scaleFactor)

    return s0, s1, s2
