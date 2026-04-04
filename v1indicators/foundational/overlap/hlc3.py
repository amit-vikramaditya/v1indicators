import pandas as pd

from .._utils import check_series


def hlc3(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """HLC3 = (high + low + close) / 3."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    out = (high_s + low_s + close_s) / 3.0
    out.name = "HLC3"
    return out
