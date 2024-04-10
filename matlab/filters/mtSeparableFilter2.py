import numpy as np
from scipy.signal import convolve2d


def mtSeparableFilter2(xFilter, yFilter, input, mode="same"):
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
    xFilter = xFilter.reshape(1, -1)
    yFilter = yFilter.reshape(-1, 1)

    if mode == "cyclic":
        # If cyclic mode selected, extend image in x (for x filter pass) and y
        # (for y filter pass) and fill extended padding with pixels from opposite
        # side of image.
        xPad = (len(xFilter) - 1) // 2
        yPad = (len(yFilter) - 1) // 2
        xPadded = np.hstack([input[:, -xPad:], input, input[:, :xPad]])
        yPadded = np.vstack([xPadded[-yPad:, :], xPadded, xPadded[:yPad, :]])
        output = convolve2d(yPadded, yFilter, mode="valid")

    elif mode == "mirror":
        # If mirror mode selected, extend image in x (for x filter pass) and y
        # (for y filter pass) and fill extended padding with reflected boundary
        # pixels.
        xPad = (len(xFilter) - 1) // 2
        yPad = (len(yFilter) - 1) // 2
        xPadded = np.hstack([input[:, xPad:0:-1], input, input[:, -2 : -xPad - 2 : -1]])
        yPadded = np.vstack(
            [xPadded[yPad:0:-1, :], xPadded, xPadded[-2 : -yPad - 2 : -1, :]]
        )
        output = convolve2d(yPadded, yFilter, mode="valid")

    else:
        # Otherwise, just apply convolve2d for x and y filters, passing selected mode
        output = convolve2d(convolve2d(input, xFilter, mode=mode), yFilter, mode=mode)

    return output
