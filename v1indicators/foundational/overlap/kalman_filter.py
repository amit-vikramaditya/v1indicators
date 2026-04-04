import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _kalman_filter_kernel(
    src_v: np.ndarray,
    high_v: np.ndarray,
    low_v: np.ndarray,
    close_v: np.ndarray,
    velocity_alpha: float,
    range_alpha: float,
    memory_alpha: float,
) -> np.ndarray:
    n = src_v.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return out

    value1 = 0.0
    value2 = 0.0
    out[0] = src_v[0]

    for i in range(1, n):
        if (
            np.isnan(src_v[i])
            or np.isnan(src_v[i - 1])
            or np.isnan(high_v[i])
            or np.isnan(low_v[i])
            or np.isnan(close_v[i - 1])
        ):
            out[i] = np.nan
            continue

        tr = max(
            high_v[i] - low_v[i],
            max(abs(high_v[i] - close_v[i - 1]), abs(low_v[i] - close_v[i - 1])),
        )

        value1 = velocity_alpha * (src_v[i] - src_v[i - 1]) + memory_alpha * value1
        value2 = range_alpha * tr + memory_alpha * value2

        lam = abs(value1 / value2) if value2 != 0.0 else 0.0
        alpha = (-lam * lam + np.sqrt(lam**4 + 16.0 * lam * lam)) / 8.0

        prev = out[i - 1]
        if np.isnan(prev):
            prev = src_v[i - 1]

        out[i] = alpha * src_v[i] + (1.0 - alpha) * prev

    return out


def kalman_filter(
    source: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    velocity_alpha: float = 0.2,
    range_alpha: float = 0.1,
    memory_alpha: float = 0.8,
) -> pd.Series:
    """
    Kalman-like adaptive smoothing filter inspired by Ehlers.

    Uses adaptive alpha from source velocity versus true-range envelope.
    """
    if velocity_alpha <= 0 or range_alpha <= 0 or not (0 <= memory_alpha < 1):
        raise ValueError(
            "velocity_alpha and range_alpha must be > 0, and memory_alpha must be in [0, 1)"
        )

    source_s = check_series(source, "source")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    out = _kalman_filter_kernel(
        source_s.to_numpy(dtype=np.float64),
        high_s.to_numpy(dtype=np.float64),
        low_s.to_numpy(dtype=np.float64),
        close_s.to_numpy(dtype=np.float64),
        float(velocity_alpha),
        float(range_alpha),
        float(memory_alpha),
    )

    return pd.Series(out, index=source_s.index, name="KALMAN_FILTER")
