import numpy as np
import cv2
from mtBifs import mtBifs
from utils.mtShowCaseBifs import mtShowCaseBifs


def mtDemoBifLab():
    """
    mtDemoBifLab
    Show BIFs for demonstration images
    """
    im = cv2.imread("F1.jpg")
    if im is None:
        print(f"Failed to load image ")
        return
    bifs = mtBifs(im, 2, 0.015)
    mtShowCaseBifs("car", im, bifs, np.arange(95, 121), np.arange(95, 121))


# Example usage
mtDemoBifLab()
