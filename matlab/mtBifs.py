import numpy as np
import cv2
import matplotlib.pyplot as plt
from filters.mtGaussianDerivativeFilters1d import mtGaussianDerivativeFilters1d
from filters.mtFilter2d import mtFilter2d


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
    def colourMap():
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
        plt.imshow(bifImage)

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

        plt.axis("off")
        plt.show()

    def get_snippet(self, rows, cols):
        # Creates new BIF object with BIF data for specified rows and columns

        bifSnippet = mtBifs()
        bifSnippet.Class = self.Class[rows, cols]
        bifSnippet.Vx = self.Vx[rows, cols]
        bifSnippet.Vy = self.Vy[rows, cols]

        return bifSnippet

    def roiHistograms(self, roiMask, type):
        # Calculates histograms of BIFs within a sliding window region of interest (ROI).
        #
        # INPUTS:
        # roiMask: A 2D matrix of weights indicating how much each pixel in the mask should contribute to the histogram
        #          (0 = not at all, 1 = fully). roiMask pixels with weights between 0 and 1 will partially contribute
        #          to the histogram (used to more accurately represent ROIs where edges don't lie on pixel boundaries).
        # type: Type of BIF histogram to calculate
        #       1 = BIF classes only. Bin indexes map directly to classes
        #           Bin 1: flat (class 1)
        #           Bin 2: gradient (class 2)
        #           Bin 3: dark blob (class 3)
        #           Bin 4: light blob (class 4)
        #           Bin 5: dark line (class 5)
        #           Bin 6: light line (class 6)
        #           Bin 7: saddle (class 7)
        #
        # USAGE: obj.roiHistograms(roiMask, type)

        filterMode = "padded"
        if type == 1:
            numBins = 7
            binBifClasses = list(range(1, numBins + 1))
            bifHistogram = np.zeros((self.Class.shape[0], self.Class.shape[1], numBins))
            for binIdx in range(numBins):
                bifClassMask = self.Class == binBifClasses[binIdx]
                histogramsForClass = mtFilter2d(roiMask, bifClassMask, filterMode)
                bifHistogram[:, :, binIdx] = histogramsForClass
            return bifHistogram
        else:
            raise ValueError("type must be 1 (BIF classes only)")

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

    @staticmethod
    def dtgFilterResponsesFromImage(inputImage, sigma):
        # Generate filter responses
        s0, s1, s2 = mtGaussianDerivativeFilters1d(sigma)

        # Paso 1: Suavizado Gaussiano
        imagen_suavizada = cv2.GaussianBlur(inputImage, (5, 5), 0)

        # Paso 2: Operador Sobel
        gradiente_x = cv2.Sobel(imagen_suavizada, cv2.CV_64F, 1, 0, ksize=3)
        gradiente_y = cv2.Sobel(imagen_suavizada, cv2.CV_64F, 0, 1, ksize=3)

        # Calcular el módulo del gradiente
        modulo_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)

        # Paso 3: Umbralización
        umbral, bordes = cv2.threshold(modulo_gradiente, 30, 255, cv2.THRESH_BINARY)

        # Definir el elemento estructural
        elemento_estructural = np.ones((3, 3), np.uint8)

        # Dilatar la imagen binaria
        imagen_dilatada = cv2.dilate(bordes, elemento_estructural, iterations=1)

        # Restar la imagen dilatada de la imagen binaria para obtener la frontera
        fronteras = imagen_dilatada - bordes

        L = cv2.Laplacian(fronteras, cv2.CV_64F, ksize=3)
        Lx = cv2.Laplacian(fronteras, cv2.CV_64F, ksize=3)
        Ly = cv2.Laplacian(fronteras, cv2.CV_64F, ksize=3)
        Lxx = cv2.Laplacian(fronteras, cv2.CV_64F, ksize=3)
        Lyy = cv2.Laplacian(fronteras, cv2.CV_64F, ksize=3)
        Lxy = cv2.Laplacian(fronteras, cv2.CV_64F, ksize=3)

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
        epsilon = 1e-7  # small constant
        vx1 /= norm1 + epsilon
        vy1 /= norm1 + epsilon

        vx2 = -(-Lxx + Lyy + np.sqrt((Lxx - Lyy) ** 2 + 4 * Lxy**2)) / (
            2 * Lxy + epsilon
        )
        vy2 = np.ones_like(vx2)
        norm2 = np.sqrt(vx2**2 + vy2**2)
        vx2 /= norm2 + epsilon
        vy2 /= norm2 + epsilon

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
