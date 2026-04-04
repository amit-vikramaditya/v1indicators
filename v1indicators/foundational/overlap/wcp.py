import pandas as pd

from .._utils import check_series


def wcp(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Weighted Close Price = (high + low + 2*close)/4."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    out = (high_s + low_s + 2.0 * close_s) / 4.0
    out.name = "WCP"
    return out
