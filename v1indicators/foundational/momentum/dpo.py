import pandas as pd

from .._utils import check_series


def dpo(close: pd.Series, length: int = 20) -> pd.Series:
    """Detrended Price Oscillator."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    shift = length // 2 + 1
    sma = close_s.rolling(length).mean()
    out = close_s.shift(-shift) - sma
    out.name = f"DPO_{length}"
    return out
