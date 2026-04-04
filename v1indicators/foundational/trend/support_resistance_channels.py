import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit
def _sr_channels_kernel(
    close_v: np.ndarray,
    pivot_v: np.ndarray,
    width_v: np.ndarray,
    loopback: int,
    min_strength: int,
):
    n = close_v.shape[0]
    resistance = np.full(n, np.nan, dtype=np.float64)
    support = np.full(n, np.nan, dtype=np.float64)
    strength_res = np.zeros(n, dtype=np.int32)
    strength_sup = np.zeros(n, dtype=np.int32)
    break_res = np.zeros(n, dtype=np.bool_)
    break_sup = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        w = width_v[i]
        if np.isnan(w) or w <= 0.0 or np.isnan(close_v[i]):
            continue

        start = i - loopback
        if start < 0:
            start = 0

        best_res_level = np.nan
        best_sup_level = np.nan
        best_res_strength = 0
        best_sup_strength = 0

        # Candidate pivots are recent pivots only.
        for j in range(start, i + 1):
            p = pivot_v[j]
            if np.isnan(p):
                continue

            cnt = 0
            acc = 0.0
            for k in range(start, i + 1):
                q = pivot_v[k]
                if np.isnan(q):
                    continue
                if abs(q - p) <= w:
                    cnt += 1
                    acc += q

            if cnt < min_strength:
                continue

            level = acc / cnt

            if level >= close_v[i]:
                if cnt > best_res_strength:
                    best_res_strength = cnt
                    best_res_level = level
            else:
                if cnt > best_sup_strength:
                    best_sup_strength = cnt
                    best_sup_level = level

        resistance[i] = best_res_level
        support[i] = best_sup_level
        strength_res[i] = best_res_strength
        strength_sup[i] = best_sup_strength

        if i > 0 and not np.isnan(resistance[i - 1]) and not np.isnan(close_v[i - 1]):
            if close_v[i - 1] <= resistance[i - 1] and close_v[i] > resistance[i - 1]:
                break_res[i] = True

        if i > 0 and not np.isnan(support[i - 1]) and not np.isnan(close_v[i - 1]):
            if close_v[i - 1] >= support[i - 1] and close_v[i] < support[i - 1]:
                break_sup[i] = True

    return resistance, support, strength_res, strength_sup, break_res, break_sup


def support_resistance_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    pivot_period: int = 10,
    channel_width_pct: float = 5.0,
    min_strength: int = 2,
    loopback: int = 290,
) -> pd.DataFrame:
    """
    Support/Resistance channels from pivot clustering.

    Inspired by TradingView support-resistance channel logic, adapted for
    calculation-only API: strongest resistance above and support below price.

    This is a retrospective pivot-based indicator: pivot anchors are only
    confirmed after `pivot_period` future bars.
    """
    if pivot_period <= 0:
        raise ValueError("pivot_period must be > 0")
    if channel_width_pct <= 0:
        raise ValueError("channel_width_pct must be > 0")
    if min_strength <= 0:
        raise ValueError("min_strength must be > 0")
    if loopback <= 0:
        raise ValueError("loopback must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    window = 2 * pivot_period + 1
    pivot_high = high_s.where(high_s == high_s.rolling(window).max().shift(-pivot_period))
    pivot_low = low_s.where(low_s == low_s.rolling(window).min().shift(-pivot_period))

    pivot_values = pivot_high.copy()
    pivot_values = pivot_values.where(pivot_values.notna(), pivot_low)

    range300 = (high_s.rolling(300).max() - low_s.rolling(300).min())
    width = range300 * (channel_width_pct / 100.0)

    resistance, support, strength_res, strength_sup, break_res, break_sup = _sr_channels_kernel(
        close_s.to_numpy(dtype=np.float64),
        pivot_values.to_numpy(dtype=np.float64),
        width.to_numpy(dtype=np.float64),
        int(loopback),
        int(min_strength),
    )

    return pd.DataFrame(
        {
            "PIVOT_VALUE": pivot_values,
            "SR_RESISTANCE": resistance,
            "SR_SUPPORT": support,
            "SR_RESISTANCE_STRENGTH": strength_res,
            "SR_SUPPORT_STRENGTH": strength_sup,
            "BREAK_RESISTANCE": break_res,
            "BREAK_SUPPORT": break_sup,
        },
        index=close_s.index,
    )
