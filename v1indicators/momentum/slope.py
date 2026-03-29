import pandas as pd

from .._utils import check_series


def slope(close: pd.Series, length: int = 1) -> pd.Series:
    """Slope over lookback length."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    out = (close_s - close_s.shift(length)) / float(length)
    out.name = f"SLOPE_{length}"
    return out
