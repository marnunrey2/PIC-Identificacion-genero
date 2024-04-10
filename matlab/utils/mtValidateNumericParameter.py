def mtValidateNumericParameter(param, param_name):
    """
    mtValidateNumericParameter
    Validates that a parameter is a number and throws an error if it is not.

    Parameters:
    - param: The parameter to validate
    - param_name (str): The name of the parameter

    Raises:
    - ValueError: If the parameter is empty or not numeric

    Usage: mtValidateNumericParameter(param, param_name)
    """
    if param is None:
        raise ValueError(f"{param_name} is empty")
    if not isinstance(param, (int, float)):
        raise ValueError(f"{param_name} is not a number")
