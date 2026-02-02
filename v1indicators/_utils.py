import pandas as pd
import numpy as np
from typing import Union, Tuple

def validate_series(data: pd.Series, name: str = "data") -> np.ndarray:
    """
    Validates that input is a Pandas Series and returns its numpy values (float64).
    Ensures data is numeric.
    """
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    
    # Ensure we are working with float64 for precision
    values = data.values.astype(np.float64)
    return values

def check_series(data: pd.Series, name: str = "data") -> pd.Series:
    """
    Validates that input is a Pandas Series, converts to float64, and returns the Series.
    Useful for Pandas-based kernels (like ewm).
    """
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    
    return data.astype(np.float64)

def to_series(data: np.ndarray, index: pd.Index, name: str = None) -> pd.Series:
    """Wraps numpy array back into a Pandas Series with the given index."""
    return pd.Series(data, index=index, name=name)
