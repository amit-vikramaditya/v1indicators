import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _zigzag_swings_kernel(high_v: np.ndarray, low_v: np.ndarray, length: int):
    n = high_v.shape[0]
    swing_high = np.full(n, np.nan, dtype=np.float64)
    swing_low = np.full(n, np.nan, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int8)

    if n == 0 or n < 2 * length + 1:
        return swing_high, swing_low, trend

    cur_trend = 0
    for i in range(length, n - length):
        h = high_v[i]
        low_i = low_v[i]
        if np.isnan(h) or np.isnan(low_i):
            trend[i] = cur_trend
            continue

        local_max = True
        local_min = True
        for j in range(i - length, i + length + 1):
            if np.isnan(high_v[j]) or np.isnan(low_v[j]):
                continue
            if high_v[j] > h:
                local_max = False
            if low_v[j] < low_i:
                local_min = False
            if not local_max and not local_min:
                break

        if local_max:
            swing_high[i] = h
            cur_trend = -1
        elif local_min:
            swing_low[i] = low_i
            cur_trend = 1

        trend[i] = cur_trend

    # carry trend forward
    for i in range(1, n):
        if trend[i] == 0:
            trend[i] = trend[i - 1]

    return swing_high, swing_low, trend


def zigzag_swings(
    high: pd.Series,
    low: pd.Series,
    length: int = 9,
) -> pd.DataFrame:
    """
    ZigZag swing points and directional state.

    Uses local extrema over a symmetric lookback/lookforward window.
    Swing labels are therefore retrospective and only confirmed after `length`
    future bars.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")

    swing_high, swing_low, trend = _zigzag_swings_kernel(
        high_s.to_numpy(dtype=np.float64),
        low_s.to_numpy(dtype=np.float64),
        int(length),
    )

    return pd.DataFrame(
        {
            "SWING_HIGH": swing_high,
            "SWING_LOW": swing_low,
            "ZZ_TREND": trend,
        },
        index=high_s.index,
    )
