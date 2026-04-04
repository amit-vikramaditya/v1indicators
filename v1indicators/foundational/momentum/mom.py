import pandas as pd

from .._utils import check_series


def mom(close: pd.Series, length: int = 10) -> pd.Series:
    """Momentum: close - close[length]."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    out = close_s - close_s.shift(length)
    out.name = f"MOM_{length}"
    return out
