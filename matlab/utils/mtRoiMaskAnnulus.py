import numpy as np


def mtRoiMaskAnnulus(rInner, rOuter, softFlag):
    """
    mtRoiMaskAnnulus
    Generates an annular region of interest (ROI) mask. All pixels within the mask
    are set to 1 and all pixels outside the mask are set to 0. If softFlag is set,
    edge pixels are included in proportion to the volume of the pixel in the ROI
    and have fractional values between 0 and 1. For a circular ROI, set rInner
    to zero.

    Parameters:
    - rInner (float): Inner radius of annulus. For a circular ROI, set rInner to zero.
    - rOuter (float): Outer radius of annulus
    - softFlag (bool): Sets whether to include edge pixels in proportion to the volume of
                       pixel in the ROI (i.e. to fractional values between 0 and 1).

    Returns:
    - roiMask (numpy.ndarray): Binary (softFlag = FALSE) or weighted (softFlag = TRUE) ROI mask.
    - X (numpy.ndarray): x-coordinates of pixels relative to centre of mask.
    - Y (numpy.ndarray): y-coordinates of pixels relative to centre of mask.

    Usage: roiMask, X, Y = mtRoiMaskAnnulus(rInner, rOuter, softFlag)
    """

    if rInner >= rOuter:
        raise ValueError("rInner must be less than rOuter")

    s = np.arange(-np.ceil(rOuter), np.ceil(rOuter) + 1)
    X, Y = np.meshgrid(s, s)
    rMask = np.sqrt(X**2 + Y**2)

    if softFlag:
        # "Soft" weighted mask
        # Gives gives a more gradual movement of the ROI edges than a hard binary
        # mask by including parts of pixels on the ROI border in approximate
        # proportion to the volume of the pixel within the ROI
        # NOTE: This doesn't take into account the fact that the proportion of the
        # pixel that lies within the ROI is angle dependent. However, it gives a
        # closer approximation to the analytical roi area given by
        # pi*(rOuter^2-rInner^2) than the hard binary mask.
        #
        # The rationale is that any pixel with a centre at radius r covers a range
        # [r-0.5..r+0.5]. Therefore max(0,min(1,rOuter-(rMask-0.5))) is the
        # proportion of the pixel below rOuter and max(0,min(1,(rMask+0.5)-rInner))
        # is the proportion of the pixel above rInner. The minimum of these
        # represents the proportion of the pixel within the ROI.
        roiMask = np.minimum(
            np.maximum(0, np.minimum(1, rOuter - (rMask - 0.5))),
            np.maximum(0, np.minimum(1, (rMask + 0.5) - rInner)),
        )
    else:
        # "Hard" binary mask
        roiMask = np.ones_like(rMask)
        roiMask[rMask < rInner] = 0
        roiMask[rMask >= rOuter] = 0

    return roiMask, X, Y
