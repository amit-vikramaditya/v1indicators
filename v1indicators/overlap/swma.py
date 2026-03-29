import numpy as np
import pandas as pd

from .._utils import check_series


def swma(close: pd.Series, length: int = 4) -> pd.Series:
    """Symmetric weighted moving average."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    i = np.arange(1, length + 1, dtype=np.float64)
    w = np.minimum(i, i[::-1])
    w /= w.sum()

    out = close_s.rolling(length).apply(lambda x: np.dot(x, w), raw=True)
    out.name = f"SWMA_{length}"
    return out
