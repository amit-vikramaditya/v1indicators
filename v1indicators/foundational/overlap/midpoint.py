import pandas as pd

from .._utils import check_series


def midpoint(close: pd.Series, length: int = 14) -> pd.Series:
    """Midpoint of rolling highest and lowest close."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    out = 0.5 * (close_s.rolling(length).max() + close_s.rolling(length).min())
    out.name = f"MIDPOINT_{length}"
    return out
