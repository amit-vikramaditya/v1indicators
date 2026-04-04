import pandas as pd

from .._utils import check_series


def ohlc4(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """OHLC4 = (open + high + low + close) / 4."""
    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    out = 0.25 * (open_s + high_s + low_s + close_s)
    out.name = "OHLC4"
    return out
