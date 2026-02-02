import pandas as pd
import numpy as np
from .._utils import check_series
from ..overlap.rma import rma

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    A momentum oscillator that measures the speed and change of price movements
    using Wilder's smoothing (RMA).

    Formula:
        RSI = 100 - [100 / (1 + RS)]
        RS = AvgGain / AvgLoss

    Args:
        close: Pandas Series of prices.
        length: Period length (default 14).

    Returns:
        Pandas Series named 'RSI_{length}'.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close = check_series(close, "close")
    delta = close.diff()

    # Note: We want to use NumPy for speed where possible, but preserving index is nice.
    # But since rma takes series, we keep series.
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use centralized RMA (Wilder's Smoothing)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)

    # Calculate RS
    # Handle division by zero if loss is 0
    # avg_loss can be 0.
    
    rs = avg_gain / avg_loss
    
    # RSI formula
    rsi_series = 100 - (100 / (1 + rs))
    
    # Handle the case where avg_loss is 0 -> RSI is 100
    # If avg_loss is 0, rs is inf. 100/(1+inf) -> 0. 100-0 -> 100.
    # But if avg_gain is also 0? 0/0 -> NaN.
    
    # If both are 0, RSI should probably be previous RSI or 50?
    # Standard behavior:
    # If Loss=0, RSI=100.
    # If Gain=0, RSI=0.
    # If Both=0 (Price flat), RSI=50? or prev value?
    # Let's clean up infs.
    
    rsi_series = rsi_series.replace([np.inf], 100.0)
    
    rsi_series.name = f"RSI_{length}"
    return rsi_series

