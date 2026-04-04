import numpy as np
import pandas as pd

from .._utils import check_series


def bias(close: pd.Series, length: int = 26) -> pd.Series:
    """BIAS = 100 * (close - SMA(close, length)) / SMA(close, length)."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    basis = close_s.rolling(length).mean().replace(0.0, np.nan)

    out = 100.0 * (close_s - basis) / basis
    out.name = f"BIAS_{length}"
    return out
