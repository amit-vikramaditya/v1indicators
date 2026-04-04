import pandas as pd

from ..._utils import check_series


def skew(close: pd.Series, length: int = 30) -> pd.Series:
    """Rolling skewness."""
    if length <= 0:
        raise ValueError("length must be > 0")
    close_s = check_series(close, "close")
    out = close_s.rolling(length).skew()
    out.name = f"SKEW_{length}"
    return out
