import numpy as np
import pandas as pd

from ..._utils import check_series


def zscore(close: pd.Series, length: int = 30, ddof: int = 0) -> pd.Series:
    """Rolling z-score: (x - mean) / stdev."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if ddof < 0:
        raise ValueError("ddof must be >= 0")

    close_s = check_series(close, "close")
    mean = close_s.rolling(length).mean()
    std = close_s.rolling(length).std(ddof=ddof).replace(0.0, np.nan)
    out = (close_s - mean) / std
    out.name = f"ZSCORE_{length}"
    return out
