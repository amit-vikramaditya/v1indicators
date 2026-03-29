import pandas as pd

from .._utils import check_series


def mad(close: pd.Series, length: int = 14) -> pd.Series:
    """Mean Absolute Deviation from rolling mean."""
    if length <= 0:
        raise ValueError("length must be > 0")
    close_s = check_series(close, "close")

    mean = close_s.rolling(length).mean()
    out = (close_s - mean).abs().rolling(length).mean()
    out.name = f"MAD_{length}"
    return out
