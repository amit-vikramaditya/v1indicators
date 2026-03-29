import numpy as np
import pandas as pd

from .._utils import check_series


def linreg(close: pd.Series, length: int = 14, offset: int = 0) -> pd.Series:
    """Rolling linear regression value at current bar."""
    if length <= 1:
        raise ValueError("length must be > 1")

    close_s = check_series(close, "close")
    x = np.arange(length, dtype=np.float64)
    x_mean = x.mean()
    var_x = ((x - x_mean) ** 2).sum()

    def _fit(y: np.ndarray) -> float:
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / var_x
        intercept = y_mean - slope * x_mean
        return intercept + slope * (length - 1 + offset)

    out = close_s.rolling(length).apply(_fit, raw=True)
    out.name = f"LINREG_{length}"
    return out
