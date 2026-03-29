import pandas as pd

from .._utils import check_series
from ..volatility.rvi import rvi


def inertia(close: pd.Series, length: int = 20, rvi_length: int = 14) -> pd.Series:
    """Inertia as smoothed RVI."""
    if length <= 0 or rvi_length <= 0:
        raise ValueError("length and rvi_length must be > 0")

    close_s = check_series(close, "close")
    rv = rvi(close_s, length=rvi_length)
    out = rv.rolling(length).mean()
    out.name = f"INERTIA_{length}_{rvi_length}"
    return out
