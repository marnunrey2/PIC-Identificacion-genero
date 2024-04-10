import os


def mtValidateFilePathParameter(param, root_dir=None):
    """
    mtValidateFilePathParameter
    Validates that a parameter is a valid path to an existing file and throws an
    error if it is not.

    Parameters:
    - param (str): The parameter to validate (file path)
    - root_dir (str, optional): The root directory to prepend if the given file path
                                 is not absolute. If not provided, the current working
                                 directory is used.

    Raises:
    - ValueError: If the parameter is empty, not a string, or the file does not exist

    Usage: mtValidateFilePathParameter(param, root_dir)
    """
    param_name = "param"

    # Ensure parameter is non-empty string (prerequisite to be a file path)
    if not param:
        raise ValueError(f"{param_name} is empty.")
    if not isinstance(param, str):
        raise ValueError(f"{param_name} is not a string. Value: {param}")

    # Ensure parameter is a full file path
    if not os.path.isabs(param):
        if root_dir is None:
            root_dir = os.getcwd()
        param = os.path.join(root_dir, param)

    # Check if file exists at given path
    if not os.path.isfile(param):
        raise ValueError(f"{param_name} is not a file. Value: {param}")
