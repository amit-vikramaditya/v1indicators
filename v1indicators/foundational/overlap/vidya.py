import pandas as pd
import numpy as np
from numba import njit

from .._utils import check_series


@njit(cache=True)
def _vidya_kernel(price: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = np.empty(price.shape[0], dtype=np.float64)
    out[0] = price[0]
    for i in range(1, price.shape[0]):
        alpha = a[i]
        if np.isnan(alpha):
            alpha = 0.0
        out[i] = alpha * price[i] + (1.0 - alpha) * out[i - 1]
    return out


def vidya(close: pd.Series, length: int = 14, alpha: float = 0.2) -> pd.Series:
    """Variable Index Dynamic Average."""
    if length <= 1:
        raise ValueError("length must be > 1")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    close_s = check_series(close, "close")

    up = close_s.diff().clip(lower=0.0).rolling(length).sum()
    down = (-close_s.diff().clip(upper=0.0)).rolling(length).sum()
    cmo_abs = ((up - down) / (up + down).replace(0.0, pd.NA)).abs().fillna(0.0)
    adapt_alpha = alpha * cmo_abs

    out_arr = _vidya_kernel(close_s.to_numpy(dtype=np.float64), adapt_alpha.to_numpy(dtype=np.float64))
    out = pd.Series(out_arr, index=close_s.index)

    out.name = f"VIDYA_{length}"
    return out
