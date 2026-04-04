import pandas as pd
from .._utils import check_series

def rma(close: pd.Series, length: int) -> pd.Series:
    """
    Wilder's Smoothing (Running Moving Average).
    Equivalent to EMA with alpha = 1 / length.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    series = check_series(close, "close")
    
    # Wilder's smoothing: alpha = 1/n
    # Note: Standard ewm implementation for 'com' or 'span' differs.
    # alpha = 1/length
    
    result = series.ewm(alpha=1.0/length, adjust=False).mean()
    result.name = f"RMA_{length}"
    return result
