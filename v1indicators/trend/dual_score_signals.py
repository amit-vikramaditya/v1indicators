import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series
from ..momentum.macd import macd
from ..momentum.rsi import rsi
from ..overlap.ema import ema
from ..overlap.sma import sma
from ..trend.adx import adx
from ..volatility.atr import atr
from ..volume.vwap import vwap
from ._step_resample import _expand_group_series, _resample_last


@njit
def _trade_ladder_kernel(
    close_v: np.ndarray,
    high_v: np.ndarray,
    low_v: np.ndarray,
    trigger_buy_v: np.ndarray,
    trigger_sell_v: np.ndarray,
    atr_v: np.ndarray,
    atr_multiplier: float,
):
    n = close_v.shape[0]

    state = np.zeros(n, dtype=np.int8)
    entry = np.full(n, np.nan, dtype=np.float64)
    sl = np.full(n, np.nan, dtype=np.float64)
    tp1 = np.full(n, np.nan, dtype=np.float64)
    tp2 = np.full(n, np.nan, dtype=np.float64)
    tp3 = np.full(n, np.nan, dtype=np.float64)
    tp4 = np.full(n, np.nan, dtype=np.float64)
    tp5 = np.full(n, np.nan, dtype=np.float64)

    tp1_hit = np.zeros(n, dtype=np.bool_)
    tp2_hit = np.zeros(n, dtype=np.bool_)
    tp3_hit = np.zeros(n, dtype=np.bool_)
    tp4_hit = np.zeros(n, dtype=np.bool_)
    tp5_hit = np.zeros(n, dtype=np.bool_)

    buy_signal = np.zeros(n, dtype=np.bool_)
    sell_signal = np.zeros(n, dtype=np.bool_)

    cur_state = 0
    cur_entry = np.nan
    cur_sl = np.nan
    cur_tp1 = np.nan
    cur_tp2 = np.nan
    cur_tp3 = np.nan
    cur_tp4 = np.nan
    cur_tp5 = np.nan

    cur_tp1_hit = False
    cur_tp2_hit = False
    cur_tp3_hit = False
    cur_tp4_hit = False
    cur_tp5_hit = False

    for i in range(n):
        if trigger_buy_v[i] and cur_state <= 0:
            cur_state = 1
            buy_signal[i] = True
            cur_entry = close_v[i]
            risk = atr_v[i] * atr_multiplier
            if np.isnan(risk) or risk < 0.0:
                risk = 0.0
            cur_sl = cur_entry - risk
            cur_tp1 = cur_entry + risk
            cur_tp2 = cur_entry + risk * 2.0
            cur_tp3 = cur_entry + risk * 3.0
            cur_tp4 = cur_entry + risk * 4.0
            cur_tp5 = cur_entry + risk * 5.0
            cur_tp1_hit = False
            cur_tp2_hit = False
            cur_tp3_hit = False
            cur_tp4_hit = False
            cur_tp5_hit = False
        elif trigger_sell_v[i] and cur_state >= 0:
            cur_state = -1
            sell_signal[i] = True
            cur_entry = close_v[i]
            risk = atr_v[i] * atr_multiplier
            if np.isnan(risk) or risk < 0.0:
                risk = 0.0
            cur_sl = cur_entry + risk
            cur_tp1 = cur_entry - risk
            cur_tp2 = cur_entry - risk * 2.0
            cur_tp3 = cur_entry - risk * 3.0
            cur_tp4 = cur_entry - risk * 4.0
            cur_tp5 = cur_entry - risk * 5.0
            cur_tp1_hit = False
            cur_tp2_hit = False
            cur_tp3_hit = False
            cur_tp4_hit = False
            cur_tp5_hit = False

        if cur_state == 1:
            if not np.isnan(cur_tp1) and high_v[i] >= cur_tp1:
                cur_tp1_hit = True
            if not np.isnan(cur_tp2) and high_v[i] >= cur_tp2:
                cur_tp2_hit = True
            if not np.isnan(cur_tp3) and high_v[i] >= cur_tp3:
                cur_tp3_hit = True
            if not np.isnan(cur_tp4) and high_v[i] >= cur_tp4:
                cur_tp4_hit = True
            if not np.isnan(cur_tp5) and high_v[i] >= cur_tp5:
                cur_tp5_hit = True
        elif cur_state == -1:
            if not np.isnan(cur_tp1) and low_v[i] <= cur_tp1:
                cur_tp1_hit = True
            if not np.isnan(cur_tp2) and low_v[i] <= cur_tp2:
                cur_tp2_hit = True
            if not np.isnan(cur_tp3) and low_v[i] <= cur_tp3:
                cur_tp3_hit = True
            if not np.isnan(cur_tp4) and low_v[i] <= cur_tp4:
                cur_tp4_hit = True
            if not np.isnan(cur_tp5) and low_v[i] <= cur_tp5:
                cur_tp5_hit = True

        state[i] = cur_state
        entry[i] = cur_entry
        sl[i] = cur_sl
        tp1[i] = cur_tp1
        tp2[i] = cur_tp2
        tp3[i] = cur_tp3
        tp4[i] = cur_tp4
        tp5[i] = cur_tp5

        tp1_hit[i] = cur_tp1_hit
        tp2_hit[i] = cur_tp2_hit
        tp3_hit[i] = cur_tp3_hit
        tp4_hit[i] = cur_tp4_hit
        tp5_hit[i] = cur_tp5_hit

    return (
        state,
        buy_signal,
        sell_signal,
        entry,
        sl,
        tp1,
        tp2,
        tp3,
        tp4,
        tp5,
        tp1_hit,
        tp2_hit,
        tp3_hit,
        tp4_hit,
        tp5_hit,
    )


def dual_score_signals(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    ema_fast: int = 9,
    ema_slow: int = 21,
    atr_length: int = 14,
    adx_length: int = 14,
    rsi_length: int = 14,
    mtf_step: int = 5,
    volume_length: int = 20,
    atr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Dual-score EMA crossover system with ATR ladder targets.

    Inspired by TradingView file 2, reworked into a native reusable indicator.
    """
    if ema_fast <= 0 or ema_slow <= 0:
        raise ValueError("ema_fast and ema_slow must be > 0")
    if atr_length <= 0 or adx_length <= 0 or rsi_length <= 0:
        raise ValueError("atr_length, adx_length, and rsi_length must be > 0")
    if mtf_step <= 0:
        raise ValueError("mtf_step must be > 0")
    if volume_length <= 0:
        raise ValueError("volume_length must be > 0")
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be > 0")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    ema_fast_s = ema(close_s, ema_fast)
    ema_slow_s = ema(close_s, ema_slow)
    atr_s = atr(high_s, low_s, close_s, length=atr_length)
    rsi_s = rsi(close_s, length=rsi_length)
    vwap_s = vwap(high_s, low_s, close_s, volume_s)
    macd_df = macd(close_s)
    adx_df = adx(high_s, low_s, close_s, length=adx_length)
    volume_avg = sma(volume_s, volume_length)

    reduced_close, groups = _resample_last(close_s, mtf_step)
    rsi_step_reduced = rsi(reduced_close, length=rsi_length)
    rsi_step = _expand_group_series(rsi_step_reduced, groups, close_s.index, name="DSS_RSI_STEP")

    adx_col = f"ADX_{adx_length}"

    bull_score = (
        (close_s > vwap_s).astype(np.float64)
        + (rsi_s > 50.0).astype(np.float64)
        + (macd_df["MACD"] > macd_df["MACD_SIGNAL"]).astype(np.float64)
        + (ema_fast_s > ema_slow_s).astype(np.float64)
        + ((adx_df[adx_col] > 25.0) & (close_s > ema_fast_s)).astype(np.float64)
        + ((volume_s > volume_avg) & (close_s > open_s)).astype(np.float64)
        + (rsi_step > 50.0).astype(np.float64)
    )
    bear_score = (
        (close_s < vwap_s).astype(np.float64)
        + (rsi_s < 50.0).astype(np.float64)
        + (macd_df["MACD"] < macd_df["MACD_SIGNAL"]).astype(np.float64)
        + (ema_fast_s < ema_slow_s).astype(np.float64)
        + ((adx_df[adx_col] > 25.0) & (close_s < ema_fast_s)).astype(np.float64)
        + ((volume_s > volume_avg) & (close_s < open_s)).astype(np.float64)
        + (rsi_step < 50.0).astype(np.float64)
    )

    bull_pct = (bull_score / 7.0) * 100.0
    bear_pct = (bear_score / 7.0) * 100.0

    diff = bull_pct - bear_pct
    bias = pd.Series(np.zeros(len(diff), dtype=np.int8), index=close_s.index)
    bias = bias.mask(diff >= 40.0, 2)
    bias = bias.mask(diff <= -40.0, -2)
    bias = bias.mask((diff > 0.0) & (diff < 40.0), 1)
    bias = bias.mask((diff < 0.0) & (diff > -40.0), -1)
    bias = bias.fillna(0).astype(np.int8)

    buy_cross = (ema_fast_s > ema_slow_s) & (ema_fast_s.shift(1) <= ema_slow_s.shift(1))
    sell_cross = (ema_fast_s < ema_slow_s) & (ema_fast_s.shift(1) >= ema_slow_s.shift(1))

    (
        state,
        buy_signal,
        sell_signal,
        entry,
        sl,
        tp1,
        tp2,
        tp3,
        tp4,
        tp5,
        tp1_hit,
        tp2_hit,
        tp3_hit,
        tp4_hit,
        tp5_hit,
    ) = _trade_ladder_kernel(
        close_s.to_numpy(dtype=np.float64),
        high_s.to_numpy(dtype=np.float64),
        low_s.to_numpy(dtype=np.float64),
        buy_cross.fillna(False).to_numpy(dtype=np.bool_),
        sell_cross.fillna(False).to_numpy(dtype=np.bool_),
        atr_s.to_numpy(dtype=np.float64),
        float(atr_multiplier),
    )

    state_s = pd.Series(state, index=close_s.index, name="DSS_STATE")
    retest = (
        ((state_s == 1) & (low_s <= ema_fast_s) & (low_s > ema_slow_s))
        | ((state_s == -1) & (high_s >= ema_fast_s) & (high_s < ema_slow_s))
    ).fillna(False)

    return pd.DataFrame(
        {
            "DSS_EMA_FAST": ema_fast_s,
            "DSS_EMA_SLOW": ema_slow_s,
            "DSS_VWAP": vwap_s,
            "DSS_ATR": atr_s,
            "DSS_RSI": rsi_s,
            "DSS_RSI_STEP": rsi_step,
            "DSS_MACD": macd_df["MACD"],
            "DSS_MACD_SIGNAL": macd_df["MACD_SIGNAL"],
            "DSS_ADX": adx_df[adx_col],
            "DSS_BULL_SCORE": bull_score,
            "DSS_BEAR_SCORE": bear_score,
            "DSS_BULL_PCT": bull_pct,
            "DSS_BEAR_PCT": bear_pct,
            "DSS_BIAS": bias,
            "DSS_BUY": pd.Series(buy_signal, index=close_s.index),
            "DSS_SELL": pd.Series(sell_signal, index=close_s.index),
            "DSS_STATE": state_s,
            "DSS_ENTRY": pd.Series(entry, index=close_s.index),
            "DSS_SL": pd.Series(sl, index=close_s.index),
            "DSS_TP1": pd.Series(tp1, index=close_s.index),
            "DSS_TP2": pd.Series(tp2, index=close_s.index),
            "DSS_TP3": pd.Series(tp3, index=close_s.index),
            "DSS_TP4": pd.Series(tp4, index=close_s.index),
            "DSS_TP5": pd.Series(tp5, index=close_s.index),
            "DSS_TP1_HIT": pd.Series(tp1_hit, index=close_s.index),
            "DSS_TP2_HIT": pd.Series(tp2_hit, index=close_s.index),
            "DSS_TP3_HIT": pd.Series(tp3_hit, index=close_s.index),
            "DSS_TP4_HIT": pd.Series(tp4_hit, index=close_s.index),
            "DSS_TP5_HIT": pd.Series(tp5_hit, index=close_s.index),
            "DSS_RETEST": retest,
        },
        index=close_s.index,
    )
