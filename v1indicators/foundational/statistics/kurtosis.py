import pandas as pd

from ..._utils import check_series


def kurtosis(close: pd.Series, length: int = 30) -> pd.Series:
    """Rolling kurtosis."""
    if length <= 0:
        raise ValueError("length must be > 0")
    close_s = check_series(close, "close")
    out = close_s.rolling(length).kurt()
    out.name = f"KURTOSIS_{length}"
    return out
