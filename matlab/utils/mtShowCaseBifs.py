import matplotlib.pyplot as plt
import cv2
from .mtImShow import mtImShow


def mtShowCaseBifs(title_text, image, bifs, zoom_rows, zoom_cols):
    """
    mtShowCaseBifs
    Displays image and BIFs for visual inspection. Three panels are shown as follows:
    - Left: entire input
    - Middle: BIF classes for entire image
    - Right: BIF classes and orientations for the specified rows and columns

    Parameters:
    - title_text (str): Text to display in figure window
    - image (numpy.ndarray): Input image for which BIFs have been calculated
    - bifs (mtBifs object): Object with BIF classes and orientations for entire image
    - zoom_rows (list): Rows to zoom in and show BIF orientations for
    - zoom_cols (list): Columns to zoom in and show BIF orientations for

    NOTE: It is recommended to show orientations only for small image areas
    (say up to about 50x50 pixels). For larger areas, it can be hard to
    distinguish the orientations and image display features tend to get a bit slow.

    Usage: mtShowCaseBifs(title_text, image, bifs, zoom_rows, zoom_columns)
    """

    # Open figure
    plt.figure(title_text)

    # Set bounding box for zoomed area
    x_min = min(zoom_cols)
    x_max = max(zoom_cols)
    y_min = min(zoom_rows)
    y_max = max(zoom_rows)
    X = [x_min, x_min, x_max, x_max, x_min]
    Y = [y_min, y_max, y_max, y_min, y_min]

    # Show test image
    plt.subplot(1, 2, 1)
    num_dims = len(image.shape)
    if num_dims == 2:
        # Convert greyscale image to RGB image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    plt.imshow(image)
    plt.plot(X, Y, color=[0, 1, 1])
    plt.xlabel("Test image")

    # Show BIF classes for whole image
    plt.subplot(1, 2, 2)
    bif_image = bifs.show()
    plt.imshow(bif_image)
    plt.plot(X, Y, color=[0, 1, 1])
    plt.xlabel("BIF class for all pixels")

    plt.show()
