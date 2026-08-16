import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series
from ...foundational.volatility.atr import atr


@njit
def _range_filter_kernel(src_v: np.ndarray, rng_v: np.ndarray):
    n = src_v.shape[0]
    filt = np.full(n, np.nan, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int8)
    signal = np.zeros(n, dtype=np.int8)

    if n == 0:
        return filt, trend, signal

    filt[0] = src_v[0]
    for i in range(1, n):
        prev_f = filt[i - 1]
        prev_t = trend[i - 1]
        cur_src = src_v[i]
        cur_rng = rng_v[i]

        if np.isnan(prev_f) or np.isnan(cur_src) or np.isnan(cur_rng):
            filt[i] = prev_f
            trend[i] = prev_t
            signal[i] = 0
            continue

        if cur_src > prev_f + cur_rng:
            cur_f = cur_src - cur_rng
        elif cur_src < prev_f - cur_rng:
            cur_f = cur_src + cur_rng
        else:
            cur_f = prev_f

        if cur_f > prev_f:
            cur_t = 1
        elif cur_f < prev_f:
            cur_t = -1
        else:
            cur_t = prev_t

        filt[i] = cur_f
        trend[i] = cur_t
        signal[i] = cur_t if cur_t != prev_t else 0

    return filt, trend, signal


def range_filter(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    sensitivity: int = 6,
    atr_length: int = 14,
    atr_multiplier: float = 0.8,
    mamode: str = "ema",
) -> pd.DataFrame:
    """ATR-scaled recursive range filter with trend state.

    A hysteresis band filter: the smoothed value steps by the band width only
    when price moves beyond the previous value plus/minus the band, and holds
    otherwise, producing a piecewise-linear trend line and a directional
    state. Band width:

        rng = ATR(atr_length) * atr_multiplier * (sensitivity / 8)

    Strictly causal: the recursion at bar i uses only bars <= i. NaN inputs
    carry the previous state forward without emitting a signal.

    Note: this is the standalone core also used by ``range_filter_confluence``;
    the historical production variant of this filter (FLOOP) used Wilder-RMA
    ATR — pass ``mamode`` accordingly once RMA ATR is exposed (see ATR
    docstring for the smoothing-convention discussion).

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC inputs (high/low drive ATR).
    sensitivity : int
        Band scaling numerator (divided by 8), as in the original variants.
    atr_length : int
        ATR length for the band width.
    atr_multiplier : float
        ATR multiplier for the band width.
    mamode : str
        ATR smoothing mode (see ``atr``).

    Returns
    -------
    pd.DataFrame
        ``RANGE_FILTER`` (filter value), ``RANGE_FILTER_TREND`` (+1/-1/0
        directional state), ``RANGE_FILTER_SIGNAL`` (trend-change bars only).
    """
    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0")
    if atr_length <= 0:
        raise ValueError("atr_length must be > 0")
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    atr_s = atr(high_s, low_s, close_s, length=atr_length, mamode=mamode)
    range_s = atr_s * float(atr_multiplier) * (float(sensitivity) / 8.0)

    filt, trend, rf_sig = _range_filter_kernel(
        close_s.to_numpy(dtype=np.float64),
        range_s.to_numpy(dtype=np.float64),
    )

    return pd.DataFrame(
        {
            "RANGE_FILTER": filt,
            "RANGE_FILTER_TREND": trend,
            "RANGE_FILTER_SIGNAL": rf_sig,
        },
        index=close_s.index,
    )
