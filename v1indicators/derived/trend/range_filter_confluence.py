import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series
from ...foundational.overlap.ema import ema
from ...derived.trend.adx import adx
from ...foundational.volatility.atr import atr
from ...foundational.volatility.chop import chop


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


def range_filter_confluence(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    sensitivity: int = 6,
    atr_length: int = 14,
    atr_multiplier: float = 0.8,
    ema_fast_length: int = 60,
    ema_slow_length: int = 200,
    adx_length: int = 14,
    adx_threshold: float = 20.0,
    chop_length: int = 14,
    chop_threshold: float = 61.8,
    use_adx: bool = True,
    use_chop: bool = True,
    use_cooldown: bool = True,
    cooldown_bars: int = 5,
    use_ema_filter: bool = True,
) -> pd.DataFrame:
    """Adaptive range-filter confluence engine with anti-chop gating.

    Inspired by TradingView file 7, implemented as a reusable trend signal API.
    """
    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0")
    if atr_length <= 0:
        raise ValueError("atr_length must be > 0")
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be > 0")
    if ema_fast_length <= 0 or ema_slow_length <= 0:
        raise ValueError("ema_fast_length and ema_slow_length must be > 0")
    if adx_length <= 0 or chop_length <= 1:
        raise ValueError("adx_length must be > 0 and chop_length must be > 1")
    if cooldown_bars <= 0:
        raise ValueError("cooldown_bars must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    atr_s = atr(high_s, low_s, close_s, length=atr_length)
    range_s = atr_s * float(atr_multiplier) * (float(sensitivity) / 8.0)

    filt, trend, rf_sig = _range_filter_kernel(
        close_s.to_numpy(dtype=np.float64),
        range_s.to_numpy(dtype=np.float64),
    )

    trend_s = pd.Series(trend, index=close_s.index)
    rf_sig_s = pd.Series(rf_sig, index=close_s.index)

    adx_df = adx(high_s, low_s, close_s, length=adx_length)
    adx_col = f"ADX_{adx_length}"
    adx_val = adx_df[adx_col]
    adx_trending = adx_val >= adx_threshold

    chop_val = chop(high_s, low_s, close_s, length=chop_length)
    chop_clear = chop_val <= chop_threshold

    ema_fast = ema(close_s, ema_fast_length)
    ema_slow = ema(close_s, ema_slow_length)

    ema_cross_bull = ema_fast > ema_slow
    ema_cross_bear = ema_fast < ema_slow
    ema_price_above = close_s > ema_fast
    ema_price_below = close_s < ema_fast
    ema_slope_up = ema_fast > ema_fast.shift(5)
    ema_slope_down = ema_fast < ema_fast.shift(5)

    ema_all_bull = ema_cross_bull & ema_price_above & ema_slope_up
    ema_all_bear = ema_cross_bear & ema_price_below & ema_slope_down

    ema_cond1 = pd.Series(np.where(trend_s == 1, ema_cross_bull, ema_cross_bear), index=close_s.index).astype(bool)
    ema_cond2 = pd.Series(np.where(trend_s == 1, ema_price_above, ema_price_below), index=close_s.index).astype(bool)

    roc5 = close_s.pct_change(5) * 100.0
    roc10 = close_s.pct_change(10) * 100.0
    roc20 = close_s.pct_change(20) * 100.0

    mom_bull = (roc5 > 0.0) & (roc10 > 0.0) & (roc20 > 0.0)
    mom_bear = (roc5 < 0.0) & (roc10 < 0.0) & (roc20 < 0.0)
    mom_aligned = ((trend_s == 1) & mom_bull) | ((trend_s == -1) & mom_bear)
    mom_partial = ((trend_s == 1) & (roc5 > 0.0)) | ((trend_s == -1) & (roc5 < 0.0))

    score_ema = (
        ema_cond1.astype(np.float64)
        + ema_cond2.astype(np.float64)
        + mom_aligned.astype(np.float64)
        + mom_partial.astype(np.float64)
    ).clip(upper=4.0)

    range_12 = atr_s * float(atr_multiplier) * (12.0 / 8.0)
    range_16 = atr_s * float(atr_multiplier) * (16.0 / 8.0)
    _, trend_12, _ = _range_filter_kernel(close_s.to_numpy(dtype=np.float64), range_12.to_numpy(dtype=np.float64))
    _, trend_16, _ = _range_filter_kernel(close_s.to_numpy(dtype=np.float64), range_16.to_numpy(dtype=np.float64))

    trend_12_s = pd.Series(trend_12, index=close_s.index)
    trend_16_s = pd.Series(trend_16, index=close_s.index)

    score_sens = ((trend_12_s == trend_s).astype(np.float64) * 2.0) + ((trend_16_s == trend_s).astype(np.float64) * 1.0)

    atr_norm = (atr_s / close_s.replace(0.0, np.nan)) * 100.0
    atr_rank = atr_s.rolling(60, min_periods=1).apply(lambda x: 100.0 * np.sum(x <= x[-1]) / float(len(x)), raw=True)
    atr_norm_q65 = atr_norm.rolling(60, min_periods=1).quantile(0.65)
    score_vol = (atr_rank < 80.0).astype(np.float64) + (atr_norm < atr_norm_q65).astype(np.float64)

    chop_gate = (
        ((not use_adx) | adx_trending)
        & ((not use_chop) | chop_clear)
    ).fillna(False)

    ema_gate_base = pd.Series(np.full(len(close_s), not use_ema_filter), index=close_s.index)
    ema_gate = (
        ema_gate_base
        | ((trend_s == 1) & ema_all_bull)
        | ((trend_s == -1) & ema_all_bear)
    ).fillna(False)

    raw_long = (rf_sig_s == 1)
    raw_short = (rf_sig_s == -1)

    long_signal = np.zeros(len(close_s), dtype=bool)
    short_signal = np.zeros(len(close_s), dtype=bool)
    cooldown_clear = np.zeros(len(close_s), dtype=bool)
    bars_since = np.full(len(close_s), 999, dtype=np.int64)

    counter = 999
    for i in range(len(close_s)):
        counter += 1
        cd_ok = (not use_cooldown) or (counter >= cooldown_bars)
        cooldown_clear[i] = cd_ok

        if bool(raw_long.iloc[i]) and bool(chop_gate.iloc[i]) and bool(ema_gate.iloc[i]) and cd_ok:
            long_signal[i] = True
            counter = 0
        elif bool(raw_short.iloc[i]) and bool(chop_gate.iloc[i]) and bool(ema_gate.iloc[i]) and cd_ok:
            short_signal[i] = True
            counter = 0

        bars_since[i] = counter

    filtered_long = raw_long & ~pd.Series(long_signal, index=close_s.index)
    filtered_short = raw_short & ~pd.Series(short_signal, index=close_s.index)

    chop_penalty = (
        ((use_adx) & (~adx_trending)).astype(np.float64) * -1.0
        + ((use_chop) & (~chop_clear)).astype(np.float64) * -1.0
    )

    strength_raw = score_ema + score_sens + score_vol + chop_penalty
    strength = strength_raw.clip(lower=0.0, upper=14.0)

    return pd.DataFrame(
        {
            "RFC_FILTER": pd.Series(filt, index=close_s.index),
            "RFC_TREND": trend_s,
            "RFC_SIGNAL_RAW": rf_sig_s,
            "RFC_LONG_RAW": raw_long,
            "RFC_SHORT_RAW": raw_short,
            "RFC_LONG": pd.Series(long_signal, index=close_s.index),
            "RFC_SHORT": pd.Series(short_signal, index=close_s.index),
            "RFC_FILTERED_LONG": filtered_long,
            "RFC_FILTERED_SHORT": filtered_short,
            "RFC_EMA_FAST": ema_fast,
            "RFC_EMA_SLOW": ema_slow,
            "RFC_ADX": adx_val,
            "RFC_CHOP": chop_val,
            "RFC_ADX_TRENDING": adx_trending,
            "RFC_CHOP_CLEAR": chop_clear,
            "RFC_CHOP_GATE": chop_gate,
            "RFC_EMA_GATE": ema_gate,
            "RFC_COOLDOWN_CLEAR": pd.Series(cooldown_clear, index=close_s.index),
            "RFC_BARS_SINCE_SIGNAL": pd.Series(bars_since, index=close_s.index),
            "RFC_SCORE_EMA": score_ema,
            "RFC_SCORE_SENS": score_sens,
            "RFC_SCORE_VOL": score_vol,
            "RFC_SCORE_STRENGTH": strength,
            "RFC_ROC5": roc5,
            "RFC_ROC10": roc10,
            "RFC_ROC20": roc20,
        },
        index=close_s.index,
    )
