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
    # Handle division by zero safely
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
    
    # RSI formula
    rsi_series = 100 - (100 / (1 + rs))
    
    # Handle the cases:
    # 1. avg_loss is 0, avg_gain > 0  => rs is inf => rsi is 100
    # 2. avg_loss is 0, avg_gain is 0 => rs is nan => rsi is nan
    
    rsi_series = rsi_series.replace([np.inf], 100.0)
    
    rsi_series.name = f"RSI_{length}"
    return rsi_series

