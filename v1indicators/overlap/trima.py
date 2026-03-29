import pandas as pd

from .._utils import check_series


def trima(close: pd.Series, length: int = 10) -> pd.Series:
    """Triangular Moving Average."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    n1 = (length + 1) // 2
    n2 = length - n1 + 1
    out = close_s.rolling(n1).mean().rolling(n2).mean()
    out.name = f"TRIMA_{length}"
    return out
