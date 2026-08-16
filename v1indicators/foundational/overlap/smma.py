import pandas as pd

from .._utils import check_series


def smma(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Smoothed Moving Average (SMMA).

    Equivalent to Wilder-style smoothing:
    SMMA_t = (SMMA_{t-1} * (length - 1) + price_t) / length

    NaN until `length` bars have elapsed (warmup).
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    out = close_s.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    out.name = f"SMMA_{length}"
    return out
