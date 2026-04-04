import numpy as np
import pandas as pd

from .._utils import check_series


def sinwma(close: pd.Series, length: int = 14) -> pd.Series:
    """Sine Weighted Moving Average."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    i = np.arange(1, length + 1, dtype=np.float64)
    w = np.sin(i * np.pi / (length + 1.0))
    w /= w.sum()

    out = close_s.rolling(length).apply(lambda x: np.dot(x, w), raw=True)
    out.name = f"SINWMA_{length}"
    return out
