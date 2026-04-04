import numpy as np
import pandas as pd

from .._utils import check_series


def _wma_values(values: np.ndarray, length: int) -> np.ndarray:
    """Vectorized WMA values aligned to the original input length."""
    out = np.full(values.shape[0], np.nan, dtype=np.float64)

    if values.shape[0] < length:
        return out

    weights = np.arange(1, length + 1, dtype=np.float64)
    weighted = np.convolve(values, weights[::-1], mode="valid") / weights.sum()
    out[length - 1 :] = weighted
    return out


def hma(close: pd.Series, length: int = 20) -> pd.Series:
    """
    Hull Moving Average (HMA).

    HMA(length) = WMA(2 * WMA(close, length/2) - WMA(close, length), sqrt(length))
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    values = close_s.to_numpy(dtype=np.float64)

    half_length = max(length // 2, 1)
    sqrt_length = max(int(np.sqrt(length)), 1)

    wma_half = _wma_values(values, half_length)
    wma_full = _wma_values(values, length)

    raw = 2.0 * wma_half - wma_full
    hma_values = _wma_values(raw, sqrt_length)

    return pd.Series(hma_values, index=close_s.index, name=f"HMA_{length}")
