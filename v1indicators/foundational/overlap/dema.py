import pandas as pd

from .._utils import check_series


def dema(close: pd.Series, length: int = 20) -> pd.Series:
    """
    Double Exponential Moving Average (DEMA).

    DEMA = 2 * EMA(close, length) - EMA(EMA(close, length), length)
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    ema1 = close_s.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()

    result = 2.0 * ema1 - ema2
    result.name = f"DEMA_{length}"
    return result
