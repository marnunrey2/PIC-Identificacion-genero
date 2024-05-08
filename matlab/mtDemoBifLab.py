import numpy as np
import cv2
from mtBifs import mtBifs
from utils.training_ai import train_svm, test_svm


def train_oBif():
    """
    mtDemoBifLab
    Show oBIFs for demonstration images
    """
    # Define the different values of sigma and epsilon
    sigma_values = [1, 2, 4, 8, 16]
    epsilon_values = [0.1, 0.01, 0.001]

    # Define the image names
    # image_names = ["F1.jpg", "M1.jpg", "F2.jpg", "F3.jpg", "F4.jpg", "M2.jpg", "M3.jpg", "M4.jpg", "test_1.jpg", "test_2.jpg"]
    image_names = ["F1.jpg", "M1.jpg"]

    # Initialize a dictionary to store the histograms
    histograms = {}

    # Iterate over each image
    for image_name in image_names:
        # Load the image
        image = cv2.imread(image_name)
        if image is None:
            print(f"Failed to load image {image_name}")
            continue

        # Initialize a list to store the histograms for this image
        histograms[image_name] = []

        # Iterate over each combination of sigma and epsilon values
        for sigma in sigma_values:
            for epsilon in epsilon_values:
                # Generate the oBIFs for the current image
                bifs = mtBifs(image, sigma, epsilon)

                # Generate the histogram for the current image
                histogram = bifs.generate_histogram()

                # Append the histogram to the list of histograms for this image
                histograms[image_name].append(histogram)

        # Concatenate the histograms for this image into one
        histograms[image_name] = np.concatenate(histograms[image_name])

    # labels = ['F', 'F', 'F', 'F', 'M', 'M', 'M', 'M']
    # clf = train_svm(histogram_f1,histogram_f2,histogram_f3,histogram_f4, histogram_m1,histogram_m2,histogram_m3,histogram_m4, labels)
    # test_svm(clf, [histogram_test_1, histogram_test_2], ['F', 'M'])

    labels = ["F", "M"]
    clf = train_svm(histograms["F1.jpg"], histograms["M1.jpg"], labels)
    # test_svm(clf, [histogram_test_1, histogram_test_2], ['F', 'M'])
    return clf


def test_oBif(clf):
    """
    mtDemoBifLab
    Show oBIFs for demonstration images
    """
    # Define the different values of sigma and epsilon
    sigma_values = [1, 2, 4, 8, 16]
    epsilon_values = [0.1, 0.01, 0.001]

    # Define the image names
    image_names = ["F2.jpg", "M2.jpg"]

    # Initialize a dictionary to store the histograms
    histograms = {}

    # Iterate over each image
    for image_name in image_names:
        # Load the image
        image = cv2.imread(image_name)
        if image is None:
            print(f"Failed to load image {image_name}")
            continue

        # Initialize a list to store the histograms for this image
        histograms[image_name] = []

        # Iterate over each combination of sigma and epsilon values
        for sigma in sigma_values:
            for epsilon in epsilon_values:
                # Generate the oBIFs for the current image
                bifs = mtBifs(image, sigma, epsilon)

                # Generate the histogram for the current image
                histogram = bifs.generate_histogram()

                # Append the histogram to the list of histograms for this image
                histograms[image_name].append(histogram)

        # Concatenate the histograms for this image into one
        histograms[image_name] = np.concatenate(histograms[image_name])

    labels = ["F", "M"]
    test_svm(clf, [histograms["F2.jpg"], histograms["M2.jpg"]], labels)


# Example usage
clf = train_oBif()
test_oBif(clf)
