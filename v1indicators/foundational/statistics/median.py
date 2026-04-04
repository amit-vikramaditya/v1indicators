import pandas as pd

from ..._utils import check_series


def median(close: pd.Series, length: int = 14) -> pd.Series:
    """Rolling median."""
    if length <= 0:
        raise ValueError("length must be > 0")
    close_s = check_series(close, "close")
    out = close_s.rolling(length).median()
    out.name = f"MEDIAN_{length}"
    return out
