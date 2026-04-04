import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series
from ...foundational.momentum.rsi import rsi
from ._step_resample import _group_end_mask, _resample_ohlc


@njit
def _rsi_divergence_kernel(
    rsi_v: np.ndarray,
    low_v: np.ndarray,
    high_v: np.ndarray,
    left: int,
    right: int,
):
    n = rsi_v.shape[0]
    pivot_low = np.zeros(n, dtype=np.bool_)
    pivot_high = np.zeros(n, dtype=np.bool_)
    bull_div = np.zeros(n, dtype=np.bool_)
    bear_div = np.zeros(n, dtype=np.bool_)

    has_prev_low = False
    prev_low_price = np.nan
    prev_low_rsi = np.nan

    has_prev_high = False
    prev_high_price = np.nan
    prev_high_rsi = np.nan

    for i in range(left, n - right):
        rv = rsi_v[i]
        if np.isnan(rv):
            continue

        is_low = True
        is_high = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            rj = rsi_v[j]
            if np.isnan(rj):
                continue
            if rj < rv:
                is_low = False
            if rj > rv:
                is_high = False
            if not is_low and not is_high:
                break

        if is_low:
            pivot_low[i] = True
            if has_prev_low and not np.isnan(low_v[i]) and not np.isnan(prev_low_price):
                if low_v[i] < prev_low_price and rv > prev_low_rsi:
                    bull_div[i] = True
            prev_low_price = low_v[i]
            prev_low_rsi = rv
            has_prev_low = True

        if is_high:
            pivot_high[i] = True
            if has_prev_high and not np.isnan(high_v[i]) and not np.isnan(prev_high_price):
                if high_v[i] > prev_high_price and rv < prev_high_rsi:
                    bear_div[i] = True
            prev_high_price = high_v[i]
            prev_high_rsi = rv
            has_prev_high = True

    return pivot_low, pivot_high, bull_div, bear_div


def htf_reversal_divergence(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    htf_step: int = 15,
    rsi_length: int = 14,
    pivot_left: int = 5,
    pivot_right: int = 5,
) -> pd.DataFrame:
    """HTF reversal-pattern flags with RSI pivot divergence confirmation.

    Inspired by TradingView file 6, focused on non-visual signal outputs.
    """
    if htf_step <= 0:
        raise ValueError("htf_step must be > 0")
    if rsi_length <= 0:
        raise ValueError("rsi_length must be > 0")
    if pivot_left <= 0 or pivot_right <= 0:
        raise ValueError("pivot_left and pivot_right must be > 0")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    h_o, h_h, h_l, h_c, groups = _resample_ohlc(open_s, high_s, low_s, close_s, htf_step)

    prev_o = h_o.shift(1)
    prev_c = h_c.shift(1)

    range_h = (h_h - h_l).replace(0.0, np.nan)
    body = (h_c - h_o).abs()
    lower_wick = pd.concat([h_o, h_c], axis=1).min(axis=1) - h_l
    upper_wick = h_h - pd.concat([h_o, h_c], axis=1).max(axis=1)

    bull_engulfing = (h_c > h_o) & (prev_c < prev_o) & (h_c >= prev_o) & (h_o <= prev_c)
    bear_engulfing = (h_c < h_o) & (prev_c > prev_o) & (h_c <= prev_o) & (h_o >= prev_c)
    hammer = (lower_wick > range_h * 0.6) & (body < range_h * 0.3)
    shooting_star = (upper_wick > range_h * 0.6) & (body < range_h * 0.3)

    end_mask = _group_end_mask(len(close_s), htf_step)
    end_idx = np.where(end_mask)[0]

    bull_arr = np.zeros(len(close_s), dtype=np.bool_)
    bear_arr = np.zeros(len(close_s), dtype=np.bool_)
    hammer_arr = np.zeros(len(close_s), dtype=np.bool_)
    star_arr = np.zeros(len(close_s), dtype=np.bool_)

    bull_arr[end_idx] = bull_engulfing.fillna(False).to_numpy(dtype=np.bool_)
    bear_arr[end_idx] = bear_engulfing.fillna(False).to_numpy(dtype=np.bool_)
    hammer_arr[end_idx] = hammer.fillna(False).to_numpy(dtype=np.bool_)
    star_arr[end_idx] = shooting_star.fillna(False).to_numpy(dtype=np.bool_)

    bull_engulfing_full = pd.Series(bull_arr, index=close_s.index)
    bear_engulfing_full = pd.Series(bear_arr, index=close_s.index)
    hammer_full = pd.Series(hammer_arr, index=close_s.index)
    star_full = pd.Series(star_arr, index=close_s.index)

    rsi_s = rsi(close_s, length=rsi_length)
    pivot_low, pivot_high, bull_div, bear_div = _rsi_divergence_kernel(
        rsi_s.to_numpy(dtype=np.float64),
        low_s.to_numpy(dtype=np.float64),
        high_s.to_numpy(dtype=np.float64),
        int(pivot_left),
        int(pivot_right),
    )

    return pd.DataFrame(
        {
            "HRD_GROUP_END": pd.Series(end_mask, index=close_s.index),
            "HRD_BULL_ENGULFING": bull_engulfing_full,
            "HRD_BEAR_ENGULFING": bear_engulfing_full,
            "HRD_HAMMER": hammer_full,
            "HRD_SHOOTING_STAR": star_full,
            "HRD_BULL_PATTERN": bull_engulfing_full | hammer_full,
            "HRD_BEAR_PATTERN": bear_engulfing_full | star_full,
            "HRD_RSI": rsi_s,
            "HRD_RSI_PIVOT_LOW": pd.Series(pivot_low, index=close_s.index),
            "HRD_RSI_PIVOT_HIGH": pd.Series(pivot_high, index=close_s.index),
            "HRD_BULL_DIV": pd.Series(bull_div, index=close_s.index),
            "HRD_BEAR_DIV": pd.Series(bear_div, index=close_s.index),
        },
        index=close_s.index,
    )
