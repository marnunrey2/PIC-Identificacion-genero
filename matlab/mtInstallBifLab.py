import os


def mtInstallBifLab(permanentFlag=True):
    """
    mtInstallBifLab
    Adds BifLab folder and subfolders to Matlab path, either for the current session
    (permanentFlag = False) or permanently (permanentFlag = True)

    Parameters:
    - permanentFlag (bool): Optional flag to determine if altered path is saved once
                            BifLab folders have been added to it. If False, folders are
                            added to path just for the current session. If True, folders are
                            added to path permanently. Default is True.

    Usage: mtInstallBifLab(permanentFlag)
    """

    # Set permanent flag to default if not supplied
    if not isinstance(permanentFlag, bool):
        permanentFlag = True

    # Get the current directory
    baseDir = os.getcwd()

    # Add top-level BifLab folder (where this script is called from)
    os.sys.path.append(baseDir)

    # Explicitly add subfolders to avoid adding hidden folders such as .git
    subfolders = ["filters", "tests", "utils"]
    for folder in subfolders:
        os.sys.path.append(os.path.join(baseDir, folder))

    if permanentFlag:
        # Save the altered path permanently
        os.sys.path.append(os.path.join(baseDir, "filters"))
        os.sys.path.append(os.path.join(baseDir, "tests"))
        os.sys.path.append(os.path.join(baseDir, "utils"))

        # Note: Python doesn't have a direct equivalent of `savepath()` as in MATLAB,
        # but the added paths will persist within the current environment.


# Usage example:
mtInstallBifLab(permanentFlag=True)
