import numpy as np
import cv2
from mtBifs import mtBifs
from utils.mtShowCaseBifs import mtShowCaseBifs
import matplotlib.pyplot as plt


def mtDemoBifLab():
    """
    mtDemoBifLab
    Show oBIFs for demonstration images
    """
    im = cv2.imread("F1.jpg")
    if im is None:
        print(f"Failed to load image ")
        return

    # Define the different values of sigma and epsilon
    sigma_values = [1, 2, 4, 8, 16]
    epsilon_values = [0.1, 0.01, 0.001]

    # List to store all histograms
    histograms = []

    # Iterate over the different values of sigma and epsilon
    for sigma in sigma_values:
        for epsilon in epsilon_values:
            bifs = mtBifs(im, sigma, epsilon)
            histogram = bifs.generate_histogram()

            # Add the histogram to the list
            histograms.append(histogram)

    # Concatenate all histograms into a single feature vector
    feature_vector = np.concatenate(histograms)

    # Plot the final concatenated histogram
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(feature_vector)), feature_vector, color="blue")
    plt.xlabel("oBIF Bin")
    plt.ylabel("Frequency")
    plt.title("Final Concatenated oBIF Histogram")
    plt.xticks(range(len(feature_vector)))
    plt.grid(True)

    plt.show()


# Example usage
mtDemoBifLab()
