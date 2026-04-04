import pandas as pd

from ..._utils import check_series


def stdev(close: pd.Series, length: int = 30, ddof: int = 0) -> pd.Series:
    """Rolling standard deviation."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if ddof < 0:
        raise ValueError("ddof must be >= 0")

    close_s = check_series(close, "close")
    out = close_s.rolling(length).std(ddof=ddof)
    out.name = f"STDEV_{length}"
    return out
