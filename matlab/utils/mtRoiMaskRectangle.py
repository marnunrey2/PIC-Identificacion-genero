import numpy as np


def mtRoiMaskRectangle(width, height):
    """
    mtRoiMaskRectangle
    Generates a rectangular region of interest (ROI) mask. All pixels within the
    mask are set to 1 and all pixels outside the mask are set to 0. For a square
    ROI, set width and height equal.

    Parameters:
    - width (int): Width of ROI (number of columns)
    - height (int): Height of ROI (number of rows)

    Returns:
    - roiMask (numpy.ndarray): Binary ROI mask.
    - X (numpy.ndarray): x-coordinates of pixels relative to centre of mask.
    - Y (numpy.ndarray): y-coordinates of pixels relative to centre of mask.

    Usage: roiMask, X, Y = mtRoiMaskRectangle(width, height)
    """

    half_width = (width - 1) / 2
    half_height = (height - 1) / 2
    x = np.arange(-half_width, half_width + 1)
    y = np.arange(-half_height, half_height + 1)
    X, Y = np.meshgrid(x, y)

    if not np.equal(np.round(width), width) or not np.equal(np.round(height), height):
        raise ValueError("Width and height must be integers")

    # "Hard" binary mask
    roi_mask = np.ones((height, width))

    return roi_mask, X, Y
