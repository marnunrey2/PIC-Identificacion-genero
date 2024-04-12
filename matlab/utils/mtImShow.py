import numpy as np
import matplotlib.pyplot as plt
import mplcursors


def mtImShow(
    im, cmap=None, flip=True, x_range=None, y_range=None, data_aspect_ratio=None
):
    """
    mtImShow
    Replacement for imshow. Does not permit the setting of the full range of options that imshow allows. Also fixes
    what appears to be a bug in the mapping of indexed image values to the colour map for uint32 and uint64 indexed
    images immshow/image maps 0..n-1 to colormap of size n (i.e. colour of pixel is colourMap(pxVal+1,:). imshow maps
    1..n to colourmap of size n (i.e. colour of pixel is colourMap(pxVal,:). The 0..n-1 mapping seems more appropriate.
    Note that the ind2rgb function also uses the 0..n-1 mapping.

    Parameters:
    - im: The image data
    - cmap: Colormap to use
    - flip: Whether to reverse the Y axis
    - x_range: Range of X axis
    - y_range: Range of Y axis
    - data_aspect_ratio: Aspect ratio of the data

    Returns:
    - im_handle: Handle to the image plot

    Usage: im_handle = mtImShow(im, cmap, flip, x_range, y_range, data_aspect_ratio)
    """
    # If colormap supplied...
    if cmap is not None:
        # Set colormap
        plt.colormaps(cmap)

    # Display image
    if x_range is None or y_range is None:
        # No range for data supplied so no axes
        im_handle = plt.imshow(im)
        # Remove axes tick marks and labels
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
    else:
        im_handle = plt.imshow(
            im, extent=[x_range[0], x_range[1], y_range[0], y_range[1]]
        )

        # Set ticks to be outside of image
        plt.gca().tick_params(direction="out")

    if len(im.shape) == 2:
        # Built-in behaviour for imshow seems to be that the image CData is mapped CData 1..n -> Colour map indices
        # 1..n (as per documentation) for indexed images of class uint32 and uint64. However, for indexed images of
        # class uint8 or uint16, it seems CData is mapped CData 0..n-1 -> Colour map indices 1..n (as desired here
        # and as done by ind2rgb).
        # Therefore, if indexed image of class uint32 or uint64 supplied, set CData to image values + 1.
        im = np.uint64(im)
        if len(im.shape) == 2:
            mplcursors.cursor(hover=True)

    if flip == 1:
        # Reverse Y axis so image displays right way up
        plt.gca().invert_yaxis()
    else:
        plt.gca().set_yticks([])

    # Set plot box to square aspect ratio (makes pixels square but doesn't make axis square if image is not)
    if data_aspect_ratio is None:
        plt.gca().set_aspect("equal", adjustable="box")
    elif data_aspect_ratio != "auto":
        plt.gca().set_aspect(data_aspect_ratio)

    return im_handle
