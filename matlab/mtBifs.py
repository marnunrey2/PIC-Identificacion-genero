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
        sigma = float(blurWidth)
        gamma = float(flatnessThreshold)

        # Generate filter responses
        L, Lx, Ly, Lxx, Lyy, Lxy = self.dtgFilterResponsesFromImage(inputImage, sigma)

        # Generate and set BIF classes
        self.Class = self.bifClassesFromFilterResponses(
            sigma, gamma, L, Lx, Ly, Lxx, Lyy, Lxy
        )

        # Generate and set BIF orientations
        self.Vx, self.Vy = self.bifOrientationsFromFilterResponses(
            L, Lx, Ly, Lxx, Lyy, Lxy
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
        jetScore = np.zeros((*L.shape, numBifClasses))

        jetScore[:, :, 0] = gamma * L
        jetScore[:, :, 1] = sigma * np.sqrt(Lx**2 + Ly**2)

        eigVal1 = (Lxx + Lyy + np.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy**2)) / 2
        eigVal2 = (Lxx + Lyy - np.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy**2)) / 2

        jetScore[:, :, 2] = sigma**2 * (eigVal1 + eigVal2) / 2
        jetScore[:, :, 3] = -(sigma**2) * (eigVal1 + eigVal2) / 2
        jetScore[:, :, 4] = sigma**2 * eigVal1 / np.sqrt(2)
        jetScore[:, :, 5] = -(sigma**2) * eigVal2 / np.sqrt(2)
        jetScore[:, :, 6] = sigma**2 * (eigVal1 - eigVal2) / 2

        return np.argmax(jetScore, axis=2) + 1

    def bifOrientationsFromFilterResponses(self, L, Lx, Ly, Lxx, Lyy, Lxy):
        # Compute unit vector orientations
        vx1, vy1 = Lx.astype("float64"), Ly.astype("float64")
        norm1 = np.sqrt(vx1**2 + vy1**2)
        vx1 /= norm1
        vy1 /= norm1

        vx2 = -(-Lxx + Lyy + np.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy**2)) / (2 * Lxy)
        vy2 = np.ones_like(vx2)
        norm2 = np.sqrt(vx2**2 + vy2**2)
        vx2 /= norm2
        vy2 /= norm2

        # Handle cases where Lxy is zero
        fudgeMask = Lxy == 0
        vertMask = Lxx > Lyy
        horzMask = Lyy > Lxx
        vx2[fudgeMask & vertMask] = 0
        vy2[fudgeMask & vertMask] = 1
        vx2[fudgeMask & horzMask] = 1
        vy2[fudgeMask & horzMask] = 0

        lightMask2 = self.Class == 6
        vxOld = vx2.copy()
        vy2Old = vy2.copy()
        vx2[lightMask2] = vy2Old[lightMask2]
        vy2[lightMask2] = -vxOld[lightMask2]

        # Copy appropriate order orientations into output variables
        vx = np.zeros_like(self.Class)
        vy = np.zeros_like(self.Class)
        mask1 = self.Class == 2
        vx[mask1] = vx1[mask1]
        vy[mask1] = vy1[mask1]
        mask2 = (self.Class >= 5) & (self.Class <= 7)
        vx[mask2] = vx2[mask2]
        vy[mask2] = vy2[mask2]

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
        #               1 = falt (pink)
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
    def drawBifDir2d(x, y, xScale, yScale, lineWidth, bifClass, vx, vy):
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
