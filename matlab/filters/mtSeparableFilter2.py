import numpy as np
from scipy.signal import convolve2d


def mtSeparableFilter2(xFilter, yFilter, input, mode="mirror"):
    """
    mtSeparableFilter2
    Performs 2D filtering of a 2D matrix input by a separable 2D filter by
    filtering by the x and y filter components in turn and combining the results

    Parameters:
    - xFilter (1D array): 1D filter comprising the x-component of a separable 2D filter
    - yFilter (1D array): 1D filter comprising the y-component of a separable 2D filter
    - input (2D array): The 2D input matrix to be filtered
    - mode (str): The filter mode. Options are:
        - 'cyclic': Returns an output the same size and the input. The input is
                    extended in x and y prior to filtering, with the extended boundaries
                    filled with pixels "wrapped round" from the opposite edge of the input
        - 'mirror': Returns an output the same size and the input. The input is
                    extended in x and y, prior to filtering, with the extended boundaries
                    filled with pixels "reflected" from the edge of the input.
        - Any valid "mode" argument to SciPy's convolve2d function. These
                    options will all extend the image in x and y prior to filtering, with the
                    extended boundaries filled according to the mode.
                    For more information, see the documentation of SciPy's convolve2d function.

    Returns:
    - output (2D array): Filtered output (same size as input)
    """
    # Check x and y filters are 1D vectors
    if xFilter.ndim != 1 or yFilter.ndim != 1:
        raise ValueError("x and y filters must be 1D vectors")
    # Check x and y filters have an odd length
    if len(xFilter) % 2 == 0 or len(yFilter) % 2 == 0:
        raise ValueError("x and y filters must be of odd length")

    # Ensure x is a row vector and y is a column vector
    xFilter = np.reshape(xFilter, (1, len(xFilter)))
    yFilter = np.reshape(yFilter, (len(yFilter), 1))

    if mode.lower() == "cyclic":
        # If cyclic mode selected, extend image in x (for x filter pass) and y (for y filter pass)
        # and fill extended padding with pixels from opposite side of image.
        # Using convolve2d in 'valid' mode on the padded image returns an output with the same dimensions as the unpadded input image.
        xOut = convolve2d(
            input, np.flip(xFilter, axis=1), mode="valid", boundary="wrap"
        )
        output = convolve2d(
            xOut, np.flip(yFilter, axis=0), mode="valid", boundary="wrap"
        )
    elif mode.lower() == "mirror":
        # If mirror mode selected, extend image in x (for x filter pass) and y (for y filter pass)
        # and fill extended padding with reflected boundary pixels.
        # Using convolve2d in 'valid' mode on the padded image returns an output with the same dimensions as the unpadded input image.
        xOut = convolve2d(
            input, np.flip(xFilter, axis=1), mode="valid", boundary="symm"
        )
        output = convolve2d(
            xOut, np.flip(yFilter, axis=0), mode="valid", boundary="symm"
        )
    else:
        # Otherwise just apply convolve2d for x and y filters, passing selected mode
        # Modes supported by convolve2d are:
        # 'same' - (default) returns the central part of the correlation that is the same size as the image.
        # 'valid' - returns only those parts of the correlation that are computed without the zero-padded edges, size(output) < size(image).
        # 'full' - returns the full 2-D correlation, size(output) > size(image).
        output = convolve2d(
            convolve2d(input, xFilter, mode=mode, boundary="fill"),
            yFilter,
            mode=mode,
            boundary="fill",
        )

    return output
