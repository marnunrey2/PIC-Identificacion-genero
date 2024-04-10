from mtIsEven import mtIsEven


def mtIsOdd(number):
    """
    mtIsOdd
    Checks whether a number is odd

    Parameters:
    - number (int): The number to check

    Returns:
    - isOdd (bool): True if the number is odd, False otherwise

    Usage: isOdd = mtIsOdd(number)
    """
    return not mtIsEven(number)
