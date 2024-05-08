import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


def train_svm(histogram_f1, histogram_m1, labels):
    """
    Trains an SVM using the provided histogram and labels.
    Args:
        histogram_f (numpy.ndarray): The oBIF histogram for female image.
        histogram_m (numpy.ndarray): The oBIF histogram for male image.
        labels (list): The corresponding labels.
    Returns:
        sklearn.svm.SVC: The trained SVM model.
    """
    X = np.vstack([histogram_f1, histogram_m1])
    y = labels  # use the passed labels

    # Create an SVM with RBF kernel
    clf = svm.SVC(kernel="rbf", C=10, gamma="scale")

    # Train the SVM
    clf.fit(X, y)

    # Predict the labels for the training data
    y_pred = clf.predict(X)

    # Calculate the accuracy of the model on the training data
    accuracy = accuracy_score(y, y_pred)
    print("Training accuracy:", accuracy)

    return clf


def test_svm(clf, histograms_test, labels_test):
    """
    Tests the trained SVM model on the provided test data.
    Args:
        clf (sklearn.svm.SVC): The trained SVM model.
        histograms_test (list of numpy.ndarray): The oBIF histograms for test images.
        labels_test (list): The corresponding labels for test data.
    Returns:
        float: The accuracy of the model on the test data.
    """
    histograms_test_stacked = np.vstack(histograms_test)
    y_pred = clf.predict(histograms_test_stacked)
    accuracy = accuracy_score(labels_test, y_pred)
    print("Test accuracy:", accuracy)
    return accuracy
