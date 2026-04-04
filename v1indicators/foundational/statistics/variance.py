import pandas as pd

from ..._utils import check_series


def variance(close: pd.Series, length: int = 30, ddof: int = 0) -> pd.Series:
    """Rolling variance."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if ddof < 0:
        raise ValueError("ddof must be >= 0")

    close_s = check_series(close, "close")
    out = close_s.rolling(length).var(ddof=ddof)
    out.name = f"VAR_{length}"
    return out
