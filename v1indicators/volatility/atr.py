import pandas as pd
from .._utils import check_series
from ..overlap.rma import rma

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """
    Average True Range (ATR).

    A market volatility indicator derived from the greatest of three values (True Range).
    Uses Wilder's Smoothing (RMA).

    Formula:
        TR = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = RMA(TR, length)

    Args:
        high: Pandas Series of high prices.
        low: Pandas Series of low prices.
        close: Pandas Series of close prices.
        length: Smoothing period (default 14).

    Returns:
        Pandas Series named 'ATR_{length}'.
    """
    
    if length <= 0:
        raise ValueError("length must be > 0")
        
    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")

    prev_close = close.shift()

    # Calculate True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's Smoothing
    result = rma(tr, length)
    result.name = f"ATR_{length}"
    return result

