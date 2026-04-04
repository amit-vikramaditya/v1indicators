import pandas as pd

from .._utils import check_series


def ttm_trend(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, length: int = 6) -> pd.Series:
    """TTM Trend color state approximation."""
    if length <= 0:
        raise ValueError("length must be > 0")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    basis = ((high_s + low_s + close_s) / 3.0).rolling(length).mean()
    out = (close_s > basis).astype(int) - (close_s < basis).astype(int)
    out.name = f"TTM_TREND_{length}"
    return out
