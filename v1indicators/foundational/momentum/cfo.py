import numpy as np
import pandas as pd

from .._utils import check_series


def cfo(close: pd.Series, length: int = 9, scalar: float = 100.0) -> pd.Series:
    """Chande Forecast Oscillator."""
    if length <= 1:
        raise ValueError("length must be > 1")

    close_s = check_series(close, "close")

    x = np.arange(length, dtype=np.float64)
    x_mean = x.mean()
    x_center = x - x_mean
    var_x = (x_center * x_center).sum()

    def _forecast(y: np.ndarray) -> float:
        y_mean = y.mean()
        beta = ((x_center * (y - y_mean)).sum()) / var_x
        alpha = y_mean - beta * x_mean
        return alpha + beta * (length - 1)

    reg = close_s.rolling(length).apply(_forecast, raw=True)
    out = scalar * (close_s - reg) / close_s.replace(0.0, np.nan)
    out.name = f"CFO_{length}"
    return out
