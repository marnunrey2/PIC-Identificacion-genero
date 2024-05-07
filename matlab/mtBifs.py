import numpy as np
import cv2
import matplotlib.pyplot as plt
from filters.mtGaussianDerivativeFilters1d import mtGaussianDerivativeFilters1d
from filters.mtSeparableFilter2 import mtSeparableFilter2


class mtBifs:
    def __init__(self, inputImage, blurWidth, flatnessThreshold):
        # Convert color image to grayscale if needed
        if len(inputImage.shape) == 3:
            inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        # Rename input parameters to terms used in BIF papers
        # Force numeric parameters to be floats to avoid mixing integers and floats
        self.sigma = float(blurWidth)
        self.gamma = float(flatnessThreshold)

        # Generate filter responses
        self.L, Lx, Ly, Lxx, Lyy, Lxy = self.dtgFilterResponsesFromImage(
            inputImage, self.sigma
        )

        # Generate and set BIF classes
        self.Class = self.bifClassesFromFilterResponses(
            self.sigma, self.gamma, self.L, Lx, Ly, Lxx, Lyy, Lxy
        )

        # Generate and set BIF orientations
        self.Vx, self.Vy = self.bifOrientationsFromFilterResponses(
            self.L, Lx, Ly, Lxx, Lyy, Lxy
        )

    @staticmethod
    def dtgFilterResponsesFromImage(inputImage, sigma):
        # Generate the 1D Gaussian Derivative filters used to calculate BIFs
        # s0 = zeroth order 1D filter
        # s1 = first-order 1D filter
        # s2 = second-order 1D filter
        s0, s1, s2 = mtGaussianDerivativeFilters1d(sigma)

        # Calculate 2D filter responses over the image using the 1D filters
        # Pad extended boundary so filter response is same size as input
        # image and pad boundary with reflected edge pixels
        filterMode = "mirror"
        # zeroth order filter
        L = mtSeparableFilter2(s0, s0, inputImage, filterMode)
        # first-order in x, zeroth in y
        Lx = mtSeparableFilter2(s1, s0, inputImage, filterMode)
        # first-order in y, zeroth in x
        Ly = mtSeparableFilter2(s0, s1, inputImage, filterMode)
        # second-order in x, zeroth in y
        Lxx = mtSeparableFilter2(s2, s0, inputImage, filterMode)
        # second-order in y, zeroth in x
        Lyy = mtSeparableFilter2(s0, s2, inputImage, filterMode)
        # first-order in x and y
        Lxy = mtSeparableFilter2(s1, s1, inputImage, filterMode)

        return L, Lx, Ly, Lxx, Lyy, Lxy

    @staticmethod
    def bifClassesFromFilterResponses(sigma, gamma, L, Lx, Ly, Lxx, Lyy, Lxy):
        # Compute BIF classes
        numBifClasses = 7
        numYs, numXs = L.shape
        jetScore = np.zeros((numYs, numXs, numBifClasses))

        # 1: flat (pink)
        jetScore[:, :, 0] = gamma * L
        # 2: gradient (grey)
        jetScore[:, :, 1] = sigma * np.sqrt(Lx**2 + Ly**2)

        # Second order BIFs are calculated from Hessian eigenvalues.
        # The formulation below has been chosen to be numerically stable
        # as some issues were encountered due to numerical precision
        # issues when using some other formulations.
        eigVal1 = (Lxx + Lyy + np.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy**2)) / 2
        eigVal2 = (Lxx + Lyy - np.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy**2)) / 2

        # 3: dark blob (black)
        jetScore[:, :, 2] = sigma**2 * (eigVal1 + eigVal2) / 2
        # 4: light blob (white)
        jetScore[:, :, 3] = -(sigma**2) * (eigVal1 + eigVal2) / 2
        # 5: dark line (blue)
        jetScore[:, :, 4] = sigma**2 * eigVal1 / np.sqrt(2)
        # 6: light line (yellow)
        jetScore[:, :, 5] = -(sigma**2) * eigVal2 / np.sqrt(2)
        # 7: saddle (green)
        jetScore[:, :, 6] = sigma**2 * (eigVal1 - eigVal2) / 2

        # Get maximum BIF score at each pixel (index in third dimension
        # corresponds to integer code for BIF class)
        bifClasses = np.argmax(jetScore, axis=2) + 1

        return bifClasses

    def bifOrientationsFromFilterResponses(self, L, Lx, Ly, Lxx, Lyy, Lxy):
        # Compute unit vector orientations for first order BIF classes
        vx1 = Lx
        vy1 = Ly
        norm1 = np.sqrt(vx1**2 + vy1**2)
        vx1 = np.divide(vx1, norm1, out=np.zeros_like(vx1), where=norm1 != 0)
        vy1 = np.divide(vy1, norm1, out=np.zeros_like(vy1), where=norm1 != 0)

        # Compute unit vector orientations for second order BIF classes
        vx2 = -(-Lxx + Lyy + np.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy**2)) / (2 * Lxy)
        vy2 = np.ones_like(vx2)
        norm2 = np.sqrt(vx2**2 + vy2**2)
        vx2 = np.divide(vx2, norm2, out=np.zeros_like(vx2), where=norm2 != 0)
        vy2 = np.divide(vy2, norm2, out=np.zeros_like(vy2), where=norm2 != 0)

        # Handle cases where Lxy is zero
        vertMask = Lxx > Lyy
        horzMask = Lyy > Lxx
        zero_mask = Lxy == 0
        vx2[zero_mask & vertMask] = 0
        vy2[zero_mask & vertMask] = 1
        vx2[zero_mask & horzMask] = 1
        vy2[zero_mask & horzMask] = 0

        # Handle light ridges
        lightMask2 = self.Class == 6
        vx2Old = vx2.copy()
        vy2Old = vy2.copy()
        vx2[lightMask2] = vy2Old[lightMask2]
        vy2[lightMask2] = -vx2Old[lightMask2]

        # Assign orientations to output variables
        vx = np.where(self.Class == 2, vx1, vx2)
        vy = np.where(self.Class == 2, vy1, vy2)

        # Another approach to assigning orientations
        # vx = np.zeros_like(vx1)
        # vy = np.zeros_like(vy1)

        # mask1 = self.Class == 2
        # vx[mask1] = vx1[mask1]
        # vy[mask1] = vy1[mask1]

        # mask2 = (self.Class >= 5) & (self.Class <= 7)
        # vx[mask2] = vx2[mask2]
        # vy[mask2] = vy2[mask2]

        return vx, vy

    @staticmethod
    def colourMap():
        # Generates colour map for use when displaying BIFs
        #
        # OUTPUTS:
        # colourMap: Follows the format of built-in Matlab colour maps. Row N+1 defines
        #            the RGB colour to display for matrix elements with value N. Colours
        #            as per the BIF journal papers from Crosier and Griffin with the
        #            addition of mapping for value 0 in row 1. This is not a valid BIF
        #            class but is required for a valid colour map.
        #               0 = invalid (cyan)
        #               1 = flat (pink)
        #               2 = gradient (grey)
        #               3 = dark blob (black)
        #               4 = light blob (white)
        #               5 = dark line (blue)
        #               6 = light line (yellow)
        #               7 = saddle (green)
        #
        # USAGE: colourMap = mtBifs.colourMap()

        # Define color map for displaying BIFs
        bif_cyan = [0, 0.5, 0.5]
        bif_pink = [1, 0.7, 0.7]
        bif_grey = [0.6, 0.6, 0.6]
        bif_black = [0, 0, 0]
        bif_white = [1, 1, 1]
        bif_blue = [0.1, 0.1, 1]
        bif_yellow = [0.9, 0.9, 0]
        bif_green = [0, 1, 0]

        return np.array(
            [
                bif_cyan,
                bif_pink,
                bif_grey,
                bif_black,
                bif_white,
                bif_blue,
                bif_yellow,
                bif_green,
            ]
        )

    @staticmethod
    def drawBifDir2d(self, x, y, xScale, yScale, lineWidth, bifClass, vx, vy):

        # Draw BIF orientation marks
        if bifClass in [0, 1, 3, 4]:
            return  # Do nothing. No direction associated with these BIF classes

        X1 = [x - (xScale * 0.5) * vx, x + (xScale * 0.5) * vx]
        Y1 = [y - (yScale * 0.5) * vy, y + (yScale * 0.5) * vy]

        if bifClass == 2:
            X1 = [x, x + (xScale * 0.5) * vx]
            Y1 = [y, y + (yScale * 0.5) * vy]
            X2 = [x - (xScale * 0.5) * vx, x]
            Y2 = [y - (yScale * 0.5) * vy, y]
            plt.plot(X1, Y1, color="w", linewidth=lineWidth)
            plt.plot(X2, Y2, color="k", linewidth=lineWidth)
        elif bifClass == 5:
            plt.plot(X1, Y1, color="k", linewidth=lineWidth)
        elif bifClass == 6:
            plt.plot(X1, Y1, color="w", linewidth=lineWidth)
        elif bifClass == 7:
            X2 = [x - (xScale * 0.5) * vy, x + (xScale * 0.5) * vy]
            Y2 = [y + (yScale * 0.5) * vx, y - (yScale * 0.5) * vx]
            plt.plot(X1, Y1, color="k", linewidth=lineWidth)
            plt.plot(X2, Y2, color="w", linewidth=lineWidth)

    def getSnippet(self, rows, cols):
        """
        Creates a new mtBifs object with BIF data for specified rows and columns

        Args:
            rows (slice or list or numpy.ndarray): Rows for which to get BIF data
            cols (slice or list or numpy.ndarray): Columns for which to get BIF data

        Returns:
            mtBifs: mtBifs object containing BIF data for specified rows and columns
        """
        bifSnippet = mtBifs.__new__(mtBifs)  # Create a new mtBifs object
        bifSnippet.sigma = self.sigma  # Copy sigma value
        bifSnippet.gamma = self.gamma  # Copy gamma value

        # Convert rows and cols to numpy arrays if they are not already
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        # Slice and copy relevant portions of Class, Vx, and Vy arrays
        bifSnippet.Class = self.Class[rows, cols]
        bifSnippet.Vx = self.Vx[rows, cols]
        bifSnippet.Vy = self.Vy[rows, cols]

        return bifSnippet

    def show(self, showOrientation=False):
        # Set BIF class colour map
        bifMap = self.colourMap()

        # Set all elements with invalid BIF classes to 0
        minValidBifClass = 1
        maxValidBifClass = 7
        bifClasses = self.Class.copy()
        bifClasses[
            (bifClasses < minValidBifClass) | (bifClasses > maxValidBifClass)
        ] = 0

        # Show BIF classes with colour map
        bifImage = bifMap[bifClasses]

        # Add direction marks if requested
        if showOrientation:
            numRows, numCols = self.Class.shape
            scale = 0.6
            lineWidth = 2
            for r in range(numRows):
                for c in range(numCols):
                    self.drawBifDir2d(
                        self,
                        c,
                        r,
                        scale,
                        scale,
                        lineWidth,
                        self.Class[r, c],
                        self.Vx[r, c],
                        self.Vy[r, c],
                    )

        return bifImage

    def generate_histogram(self):
        """
        Generates the oBIF histogram for the current mtBifs object.

        Returns:
            numpy.ndarray: The oBIF histogram.
        """
        # Initialize histogram bins
        num_bins = 23  # From the given formula 5n + 3
        histogram = np.zeros(num_bins)

        # Iterate over each pixel in the image
        for row in range(self.Class.shape[0]):
            for col in range(self.Class.shape[1]):
                # Determine the orientation bin index based on oBIF orientation and class
                if self.Class[row, col] == 2:  # Gray class
                    orientation = (
                        np.arctan2(self.Vy[row, col], self.Vx[row, col]) / np.pi
                    )
                    if 0.125 >= orientation > -0.125:
                        bin_index = 1  # 1st quadrant
                    elif 0.375 >= orientation > 0.125:
                        bin_index = 2  # 2nd quadrant
                    elif 0.625 >= orientation > 0.375:
                        bin_index = 3  # 3rd quadrant
                    elif 0.875 >= orientation > 0.625:
                        bin_index = 4  # 4th quadrant
                    elif -0.125 >= orientation > -0.375:
                        bin_index = 8  # 8th quadrant
                    elif -0.375 >= orientation > -0.625:
                        bin_index = 7  # 7th quadrant
                    elif -0.625 >= orientation > -0.875:
                        bin_index = 6  # 6th quadrant
                    else:
                        bin_index = 5  # 5th quadrant
                elif self.Class[row, col] == 3:  # black class
                    bin_index = 9
                elif self.Class[row, col] == 4:  # white class
                    bin_index = 10
                elif self.Class[row, col] == 5:  # Blue class
                    orientation = (
                        np.arctan2(self.Vy[row, col], self.Vx[row, col]) / np.pi
                    )
                    if 0.875 >= orientation > 0.625 or -0.125 >= orientation > -0.375:
                        bin_index = 14  # 4th and 8th quadrant
                    elif 0.625 >= orientation > 0.375 or -0.375 >= orientation > -0.625:
                        bin_index = 13  # 3th and 7th quadrant
                    elif 0.375 >= orientation > 0.125 or -0.625 >= orientation > -0.875:
                        bin_index = 12  # 2nd and 6th quadrant
                    else:
                        bin_index = 11  # 1st and 5th quadrant
                elif self.Class[row, col] == 6:  # Yellow class
                    orientation = (
                        np.arctan2(self.Vy[row, col], self.Vx[row, col]) / np.pi
                    )
                    if 0.875 >= orientation > 0.625 or -0.125 >= orientation > -0.375:
                        bin_index = 18  # 4th and 8th quadrant
                    elif 0.625 >= orientation > 0.375 or -0.375 >= orientation > -0.625:
                        bin_index = 17  # 3th and 7th quadrant
                    elif 0.375 >= orientation > 0.125 or -0.625 >= orientation > -0.875:
                        bin_index = 16  # 2nd and 6th quadrant
                    else:
                        bin_index = 15  # 1st and 5th quadrant
                elif self.Class[row, col] == 7:  # Green class
                    orientation = (
                        np.arctan2(self.Vy[row, col], self.Vx[row, col]) / np.pi
                    )
                    if 0.875 >= orientation > 0.625 or -0.125 >= orientation > -0.375:
                        bin_index = 22  # 4th and 8th quadrant
                    elif 0.625 >= orientation > 0.375 or -0.375 >= orientation > -0.625:
                        bin_index = 21  # 3th and 7th quadrant
                    elif 0.375 >= orientation > 0.125 or -0.625 >= orientation > -0.875:
                        bin_index = 20  # 2nd and 6th quadrant
                    else:
                        bin_index = 19  # 1st and 5th quadrant
                else:  # invalid classes y pink class
                    continue

                # Increment the corresponding bin in the histogram
                histogram[bin_index] += 1

        # Normalize the histogram
        histogram /= np.sum(histogram)

        return histogram
