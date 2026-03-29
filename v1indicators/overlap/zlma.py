import pandas as pd

from .._utils import check_series


def zlma(close: pd.Series, length: int = 20) -> pd.Series:
    """Zero-Lag Moving Average (EMA-based variant)."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    lag = (length - 1) // 2
    adjusted = close_s + (close_s - close_s.shift(lag))
    out = adjusted.ewm(span=length, adjust=False).mean()
    out.name = f"ZLMA_{length}"
    return out
