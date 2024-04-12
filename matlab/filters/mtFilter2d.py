import numpy as np
from scipy.signal import convolve2d


def mtFilter2d(xyFilter, input, mode, method="conv2"):
    """
    mtFilter2d
    Performs 2D filtering of a 2D matrix input by a 2D filter

    Parameters:
    - xyFilter (2D array): 2D filter. Filter must be of odd size in both dimensions (and
                            therefore have a centre pixel)
    - input (2D array): The 2D input matrix to be filtered
    - mode (str): The filter mode. Options are:
        - 'cyclic': Returns an output the same size and the input. The input is
                    extended in x and y prior to filtering, with the extended boundaries
                    filled with pixels "wrapped round" from the opposite edge of the input
        - 'mirror': Returns an output the same size as the input. The input is
                    extended in x and y, prior to filtering, with the extended boundaries
                    filled with pixels "reflected" from the edge of the input.
        - 'valid': Performs a 'valid' filter operation (i.e. no filtered values
                    for border pixels, but returns an output the same size as the input by
                    padding the valid output with NaN
    - method (str): Method to use to perform filtering operation. Optional. If not
                    provided, defaults to 'conv2'
        - 'conv2': use built-in MATLAB conv2 method (conv2 is faster than filter2
                    in some older MATLAB versions on multicore machines)
        - 'convolve2': Use third-party convolve2 function
        - 'convnfft': Use third-party convnfft function

    Returns:
    - output (2D array): Filtered output. Same size as input.

    Usage: output = mtFilter2d(xyFilter, input, mode, method)
    """
    if method not in ["conv2", "convolve2", "convnfft"]:
        raise ValueError("Invalid method argument.")

    height, width = xyFilter.shape
    # check filter is odd in both dimensions (therefore has a centre pixel)
    if (width % 2 != 1) or (height % 2 != 1):
        raise ValueError("filter must have dimensions of odd length")

    xPad = (width - 1) // 2
    yPad = (height - 1) // 2
    rawMode = "valid"
    postPadNans = False  # Used for 'padded' mode
    ## If custom mode, amend image and mode then apply filter2/conv2
    if mode.lower() == "cyclic":
        # If cyclic mode selected, extend image in x and y and fill extended padding
        # with pixels from opposite side of image. Using filter2 in 'valid' mode on the
        # padded image returns an output with the same dimensions as the unpadded input
        # image.
        input = np.hstack([input[:, -xPad:], input, input[:, :xPad]])
        input = np.vstack([input[-yPad:, :], input, input[:yPad, :]])
    elif mode.lower() == "mirror":
        # If mirror mode selected, extend image in x and y  and fill extended padding
        # with reflected boundary pixels. Using filter2 in 'valid' mode on the padded
        # image returns an output with the same dimensions as the unpadded input image.
        input = np.hstack([input[:, xPad:0:-1], input, input[:, -1 : -xPad - 1 : -1]])
        input = np.vstack([input[yPad:0:-1, :], input, input[-1 : -yPad - 1 : -1, :]])
    elif mode.lower() == "padded":
        postPadNans = True

    # Perform correlation by flipping filter and doing convolution
    if method.lower() == "conv2":
        output = convolve2d(input, xyFilter[::-1, ::-1], mode=rawMode)
    else:
        raise NotImplementedError(
            "Other convolution methods are not implemented in SciPy."
        )

    if postPadNans:
        xPadStrip = np.full((output.shape[0], xPad), np.nan)
        output = np.hstack([xPadStrip, output, xPadStrip])
        yPadStrip = np.full((yPad, output.shape[1]), np.nan)
        output = np.vstack([yPadStrip, output, yPadStrip])

    return output
