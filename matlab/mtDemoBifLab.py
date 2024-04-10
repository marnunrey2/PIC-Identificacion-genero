import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import io
from mtBifs import mtBifs
from utils.mtShowCaseBifs import mtShowCaseBifs


def mtDemoBifLab():
    """
    mtDemoBifLab
    Show BIFs for demonstration images
    """
    # car
    im = cv2.imread("F1.jpg")
    if im is None:
        print(f"Failed to load image ")
        return
    print(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    bifs = mtBifs(im, 2, 0.015)  # Assuming mtBifs function is defined elsewhere
    mtShowCaseBifs(
        "car", im, bifs, np.arange(95, 121), np.arange(95, 121)
    )  # Assuming mtShowCaseBifs function is defined elsewhere


# Example usage
mtDemoBifLab()
