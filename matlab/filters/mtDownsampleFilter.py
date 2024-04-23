import numpy as np


def mtDownsampleFilter(inFilter, scaleFactor):
    """
    mtDownsampleFilter
    Downsamples a 1D filter by averaging across scaleFactor bins.

    Parameters:
    - inFilter (1D array): Input filter to be downsampled
    - scaleFactor (int): Factor by which to downsample the input filter. Defines the
                        number of bins to average over and collapse
                        into a single bin in the output filter

    Returns:
    - outFilter (1D array): Downsampled filter

    Usage: outFilter = mtDownsampleFilter(inFilter, scaleFactor)
    """
    numBins = len(inFilter)
    outSize = numBins // scaleFactor
    outFilter = np.zeros(outSize)

    for i in range(outSize):
        outFilter[i] = np.mean(inFilter[i * scaleFactor : (i + 1) * scaleFactor])

    return outFilter
