import pandas as pd

from .._utils import check_series


def zlema(close: pd.Series, length: int = 20) -> pd.Series:
    """
    Zero-Lag Exponential Moving Average (ZLEMA).

    ZLEMA applies EMA to a lag-compensated price:
    adjusted = close + (close - close.shift(lag))
    lag = floor((length - 1) / 2)
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")

    lag = int((length - 1) / 2)
    adjusted = close_s + (close_s - close_s.shift(lag))

    result = adjusted.ewm(span=length, adjust=False).mean()
    result.name = f"ZLEMA_{length}"
    return result
