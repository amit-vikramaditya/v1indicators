import pandas as pd
import numpy as np
from .._utils import validate_series, to_series

def sma(close: pd.Series, length: int) -> pd.Series:
    """
    Simple Moving Average (SMA).

    The average of the last `n` prices. Each price has equal weight.

    Formula:
        SMA = (P1 + P2 + ... + Pn) / n

    Args:
        close: Pandas Series of prices.
        length: Number of periods for the window.

    Returns:
        Pandas Series named 'SMA_{length}'.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    # 1. Extract values
    values = validate_series(close, "close")
    
    # 2. NumPy Calculation
    # We use convolve for efficient moving average
    if len(values) < length:
        # Not enough data
        result = np.full_like(values, np.nan)
    else:
        weights = np.ones(length) / length
        # 'valid' mode returns only the parts where the window fully overlaps
        # This reduces the output size by length - 1
        sma_valid = np.convolve(values, weights, mode='valid')
        
        # Prepend NaNs to match original size
        # Expected NaNs = length - 1
        nans = np.full(length - 1, np.nan)
        result = np.concatenate((nans, sma_valid))

    # 3. Wrap result
    return to_series(result, close.index, name=f"SMA_{length}")
