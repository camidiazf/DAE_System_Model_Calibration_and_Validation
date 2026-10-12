import os
from typing import Dict, Optional, Tuple, List

import pandas as pd

# General auxiliary functions

def format_number(x: float | list | None, decimals: int = 10) -> Optional[str]:
    """
    Function to format a number to a specified number of decimal places.
    Returns None if input is None or cannot be converted to float, and formatted string otherwise.
    """

    if x is None:
        return None
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    try:
        return f"{float(x):.{decimals}f}"
    except (ValueError, TypeError):
        return x
    





