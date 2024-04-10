import numpy as np


def mtDownsampleFilter(inFilter, scaleFactor):
    """
    mtDownsampleFilter
    Downsamples 1D or 2D filters by averaging across scaleFactor bins. Generating
    a discrete filter at a higher resolution than required and then downsampling
    to the required resolution can often help produce a more accurate filter
    representation. This is especially true where filters will have a low
    resolution and it is important to preserve symmetry.

    Parameters:
    - inFilter (2D array): Input filter to be downsampled
    - scaleFactor (int): Factor by which to downsample the input filter. Defines the
                        number of bins in each dimension to average over and collapse
                        into a single bin in the output filter

    Returns:
    - outFilter (2D array): Downsampled filter

    Usage: outFilter = mtDownsampleFilter(inFilter, scaleFactor)
    """
    # Generate averaging filter
    if inFilter.ndim == 1:
        numDims = 1
    else:
        numDims = 2

    numPixelsInBin = scaleFactor**numDims
    if numDims == 1:
        sampleFilter = np.ones(scaleFactor) / numPixelsInBin
    else:
        sampleFilter = np.ones((scaleFactor, scaleFactor)) / numPixelsInBin

    # Calculate rolling window average for each bin in input filter
    rollingAverage = np.convolve(
        np.convolve(inFilter, sampleFilter, mode="valid"), sampleFilter.T, mode="valid"
    )

    if rollingAverage.ndim == 1:
        numCols = rollingAverage.shape[0]
        numRows = 1
    else:
        numRows, numCols = rollingAverage.shape

    # Determine central bin at which to start subsampling rolling average
    ctrColA = (numCols + 1) // 2
    ctrColB = numCols // 2
    if numDims == 2:
        ctrRowA = (numRows + 1) // 2
        ctrRowB = numRows // 2
    else:
        ctrRowA = 1
        ctrRowB = 1

    # Subsample rolling average to generate downsampled filter
    outFilterAA = subsample(rollingAverage, scaleFactor, ctrColA, ctrRowA)
    outFilterAB = subsample(rollingAverage, scaleFactor, ctrColA, ctrRowB)
    outFilterBA = subsample(rollingAverage, scaleFactor, ctrColB, ctrRowA)
    outFilterBB = subsample(rollingAverage, scaleFactor, ctrColB, ctrRowB)

    outFilter = (outFilterAA + outFilterAB + outFilterBA + outFilterBB) / 4

    return outFilter


def subsample(input, sampleStep, startCol, startRow):

    if input.ndim == 1:
        numCols = input.shape[0]
        numRows = 1
    else:
        numRows, numCols = input.shape

    sampleColumnIndexesFront = np.arange(startCol, 0, -sampleStep)
    sampleColumnIndexesFront = sampleColumnIndexesFront[::-1]

    sampleColumnIndexesBack = np.arange(startCol + sampleStep, numCols, sampleStep)

    sampleRowIndexesFront = np.arange(startRow, 0, -sampleStep)
    sampleRowIndexesFront = sampleRowIndexesFront[::-1]

    sampleRowIndexesBack = np.arange(startRow + sampleStep, numRows, sampleStep)

    sampleColumnIndexes = np.concatenate(
        (sampleColumnIndexesFront, sampleColumnIndexesBack)
    )
    sampleRowIndexes = np.concatenate((sampleRowIndexesFront, sampleRowIndexesBack))

    if input.ndim == 1:
        output = input[sampleColumnIndexes]
    else:
        output = input[sampleRowIndexes[:, None], sampleColumnIndexes]

    return output
