import numpy as np
import pandas as pd

from .._utils import check_series


def cg(close: pd.Series, length: int = 10) -> pd.Series:
    """Center of Gravity oscillator."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    w = np.arange(1.0, length + 1.0)
    numerator = close_s.rolling(length).apply(lambda x: -np.dot(x, w), raw=True)
    denominator = close_s.rolling(length).sum().replace(0.0, np.nan)

    out = numerator / denominator
    out.name = f"CG_{length}"
    return out
