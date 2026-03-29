import numpy as np
import pandas as pd

from .._utils import check_series


def alma(close: pd.Series, length: int = 9, sigma: float = 6.0, offset: float = 0.85) -> pd.Series:
    """Arnaud Legoux Moving Average (ALMA)."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not 0.0 <= offset <= 1.0:
        raise ValueError("offset must be between 0 and 1")

    close_s = check_series(close, "close")

    m = offset * (length - 1)
    s = length / sigma
    i = np.arange(length, dtype=np.float64)
    w = np.exp(-((i - m) ** 2) / (2.0 * s * s))
    w /= w.sum()

    out = close_s.rolling(length).apply(lambda x: np.dot(x, w), raw=True)
    out.name = f"ALMA_{length}_{sigma}_{offset}"
    return out
