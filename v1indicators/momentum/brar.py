import pandas as pd

from .._utils import check_series


def brar(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, length: int = 26) -> pd.DataFrame:
    """BRAR indicator with BR and AR lines."""
    if length <= 0:
        raise ValueError("length must be > 0")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    prev_close = close_s.shift(1)

    ar_num = (high_s - open_s).clip(lower=0.0).rolling(length).sum()
    ar_den = (open_s - low_s).clip(lower=0.0).rolling(length).sum().replace(0.0, pd.NA)
    ar = 100.0 * ar_num / ar_den

    br_num = (high_s - prev_close).clip(lower=0.0).rolling(length).sum()
    br_den = (prev_close - low_s).clip(lower=0.0).rolling(length).sum().replace(0.0, pd.NA)
    br = 100.0 * br_num / br_den

    return pd.DataFrame({f"AR_{length}": ar, f"BR_{length}": br})
