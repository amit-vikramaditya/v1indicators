import pandas as pd

from .._utils import check_series
from ..volatility.atr import atr


def pgo(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Pretty Good Oscillator = (close - SMA(close, length)) / ATR(length)."""
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    basis = close_s.rolling(length).mean()
    atr_s = atr(high_s, low_s, close_s, length=length)

    out = (close_s - basis) / atr_s
    out.name = f"PGO_{length}"
    return out
