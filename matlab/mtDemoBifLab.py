import numpy as np
import cv2
from mtBifs import mtBifs
from utils.mtShowCaseBifs import mtShowCaseBifs


def mtDemoBifLab():
    """
    mtDemoBifLab
    Show oBIFs for demonstration images
    """
    im = cv2.imread("F1.jpg")
    if im is None:
        print(f"Failed to load image ")
        return
    bifs = mtBifs(im, 2, 0.015)
    mtShowCaseBifs("car", im, bifs, np.arange(450, 510), np.arange(580, 635))
    # mtShowCaseBifs("car", im, bifs, np.arange(95, 120), np.arange(95, 120))


# Example usage
mtDemoBifLab()
