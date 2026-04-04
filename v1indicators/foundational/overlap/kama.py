import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _kama_kernel(close_v: np.ndarray, sc_v: np.ndarray, length: int) -> np.ndarray:
    out = np.full(close_v.shape[0], np.nan, dtype=np.float64)

    if close_v.shape[0] < length:
        return out

    out[length - 1] = np.mean(close_v[:length])

    for i in range(length, close_v.shape[0]):
        prev = out[i - 1]
        if np.isnan(prev) or np.isnan(close_v[i]) or np.isnan(sc_v[i]):
            out[i] = np.nan
        else:
            out[i] = prev + sc_v[i] * (close_v[i] - prev)

    return out


def kama(
    close: pd.Series,
    length: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA).
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be > 0")

    close_s = check_series(close, "close")

    change = (close_s - close_s.shift(length)).abs()
    volatility = close_s.diff().abs().rolling(length).sum().replace(0.0, np.nan)
    er = change / volatility

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama_values = _kama_kernel(
        close_s.to_numpy(dtype=np.float64),
        sc.to_numpy(dtype=np.float64),
        length,
    )

    return pd.Series(kama_values, index=close_s.index, name=f"KAMA_{length}_{fast}_{slow}")
