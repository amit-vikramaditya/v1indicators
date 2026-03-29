import math
import numpy as np
import pandas as pd

from .._utils import check_series


def pwma(close: pd.Series, length: int = 10) -> pd.Series:
    """Pascal Weighted Moving Average."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    w = np.array([math.comb(length - 1, i) for i in range(length)], dtype=np.float64)
    w /= w.sum()
    out = close_s.rolling(length).apply(lambda x: np.dot(x, w), raw=True)
    out.name = f"PWMA_{length}"
    return out
