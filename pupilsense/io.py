import os

def get_base_dir():
    """
    Returns the base directory of the project by going up two levels from the current script directory.
    """
    script_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(script_path))
    return base_dir