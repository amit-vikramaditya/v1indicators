import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit(cache=True)
def _mcgd_kernel(price: np.ndarray, length: int) -> np.ndarray:
    out = np.empty(price.shape[0], dtype=np.float64)
    out[0] = price[0]
    for i in range(1, price.shape[0]):
        prev = out[i - 1]
        denom = length * (price[i] / prev) ** 4 if prev != 0.0 else length
        out[i] = prev + (price[i] - prev) / denom
    return out


def mcgd(close: pd.Series, length: int = 14) -> pd.Series:
    """McGinley Dynamic."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    arr = _mcgd_kernel(close_s.to_numpy(dtype=np.float64), length)
    out = pd.Series(arr, index=close_s.index)
    out.name = f"MCGD_{length}"
    return out
