import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series
from ...derived.momentum.macd import macd
from ...foundational.momentum.rsi import rsi
from ...foundational.overlap.ema import ema
from ...foundational.overlap.sma import sma
from ...derived.trend.adx import adx
from ...foundational.volatility.atr import atr
from ...foundational.volume.vwap import vwap
from ._step_resample import _expand_group_series, _resample_last


_PRESETS = {
    "scalping": (5, 13, 34, 8, 10, 4, 0.8),
    "aggressive": (8, 18, 50, 11, 12, 3, 1.2),
    "default": (9, 21, 55, 13, 14, 5, 1.5),
    "conservative": (12, 26, 89, 14, 14, 7, 2.0),
    "swing": (13, 34, 89, 21, 20, 6, 2.5),
    "crypto-24-7": (9, 21, 55, 14, 20, 5, 2.0),
}


def _resolve_profile(
    preset: str,
    ema_fast: int,
    ema_slow: int,
    ema_trend: int,
    rsi_length: int,
    atr_length: int,
    min_score: int,
    sl_mult: float,
) -> tuple[int, int, int, int, int, int, float, str]:
    p = preset.strip().lower()
    if p == "auto":
        p = "default"

    if p in ("custom",):
        return ema_fast, ema_slow, ema_trend, rsi_length, atr_length, min_score, sl_mult, "custom"

    if p not in _PRESETS:
        raise ValueError(
            "preset must be one of ['auto', 'conservative', 'default', 'aggressive', "
            "'scalping', 'swing', 'crypto-24-7', 'custom']"
        )

    ef, es, et, rl, al, ms, sm = _PRESETS[p]
    return ef, es, et, rl, al, ms, sm, p


@njit
def _precision_trade_kernel(
    close_v: np.ndarray,
    high_v: np.ndarray,
    low_v: np.ndarray,
    buy_v: np.ndarray,
    sell_v: np.ndarray,
    atr_v: np.ndarray,
    sl_mult: float,
    tp1_mult: float,
    tp2_mult: float,
    tp3_mult: float,
    use_structure: bool,
    use_trailing: bool,
    swing_low_v: np.ndarray,
    swing_high_v: np.ndarray,
):
    n = close_v.shape[0]
    direction = np.zeros(n, dtype=np.int8)
    entry = np.full(n, np.nan, dtype=np.float64)
    sl = np.full(n, np.nan, dtype=np.float64)
    tp1 = np.full(n, np.nan, dtype=np.float64)
    tp2 = np.full(n, np.nan, dtype=np.float64)
    tp3 = np.full(n, np.nan, dtype=np.float64)
    trail = np.full(n, np.nan, dtype=np.float64)

    tp1_hit = np.zeros(n, dtype=np.bool_)
    tp2_hit = np.zeros(n, dtype=np.bool_)
    tp3_hit = np.zeros(n, dtype=np.bool_)
    sl_hit = np.zeros(n, dtype=np.bool_)

    cur_dir = 0
    cur_entry = np.nan
    cur_sl = np.nan
    cur_tp1 = np.nan
    cur_tp2 = np.nan
    cur_tp3 = np.nan
    cur_trail = np.nan
    cur_tp1_hit = False
    cur_tp2_hit = False
    cur_tp3_hit = False
    cur_sl_hit = False

    for i in range(n):
        if buy_v[i]:
            cur_dir = 1
            cur_entry = close_v[i]
            atr_i = atr_v[i]
            if np.isnan(atr_i) or atr_i < 0.0:
                atr_i = 0.0

            atr_stop = cur_entry - atr_i * sl_mult
            cur_sl = atr_stop

            if use_structure and not np.isnan(swing_low_v[i]):
                struct_stop = swing_low_v[i] - atr_i * 0.2
                if np.isnan(cur_sl) or struct_stop > cur_sl:
                    cur_sl = struct_stop

            min_dist = atr_i * 0.5
            if not np.isnan(cur_sl) and abs(cur_entry - cur_sl) < min_dist:
                cur_sl = cur_entry - min_dist

            risk = abs(cur_entry - cur_sl)
            cur_tp1 = cur_entry + risk * tp1_mult
            cur_tp2 = cur_entry + risk * tp2_mult
            cur_tp3 = cur_entry + risk * tp3_mult
            cur_trail = cur_sl

            cur_tp1_hit = False
            cur_tp2_hit = False
            cur_tp3_hit = False
            cur_sl_hit = False

        elif sell_v[i]:
            cur_dir = -1
            cur_entry = close_v[i]
            atr_i = atr_v[i]
            if np.isnan(atr_i) or atr_i < 0.0:
                atr_i = 0.0

            atr_stop = cur_entry + atr_i * sl_mult
            cur_sl = atr_stop

            if use_structure and not np.isnan(swing_high_v[i]):
                struct_stop = swing_high_v[i] + atr_i * 0.2
                if np.isnan(cur_sl) or struct_stop < cur_sl:
                    cur_sl = struct_stop

            min_dist = atr_i * 0.5
            if not np.isnan(cur_sl) and abs(cur_entry - cur_sl) < min_dist:
                cur_sl = cur_entry + min_dist

            risk = abs(cur_entry - cur_sl)
            cur_tp1 = cur_entry - risk * tp1_mult
            cur_tp2 = cur_entry - risk * tp2_mult
            cur_tp3 = cur_entry - risk * tp3_mult
            cur_trail = cur_sl

            cur_tp1_hit = False
            cur_tp2_hit = False
            cur_tp3_hit = False
            cur_sl_hit = False

        if cur_dir == 1 and not cur_sl_hit:
            pre_trail = cur_trail
            if not np.isnan(cur_tp1) and high_v[i] >= cur_tp1 and not cur_tp1_hit:
                cur_tp1_hit = True
                if use_trailing:
                    cur_trail = cur_entry
            if not np.isnan(cur_tp2) and high_v[i] >= cur_tp2 and not cur_tp2_hit:
                cur_tp2_hit = True
                if use_trailing:
                    cur_trail = cur_tp1
            if not np.isnan(cur_tp3) and high_v[i] >= cur_tp3 and not cur_tp3_hit:
                cur_tp3_hit = True
                if use_trailing:
                    cur_trail = cur_tp2
            if not np.isnan(pre_trail) and low_v[i] <= pre_trail:
                cur_sl_hit = True

        elif cur_dir == -1 and not cur_sl_hit:
            pre_trail = cur_trail
            if not np.isnan(cur_tp1) and low_v[i] <= cur_tp1 and not cur_tp1_hit:
                cur_tp1_hit = True
                if use_trailing:
                    cur_trail = cur_entry
            if not np.isnan(cur_tp2) and low_v[i] <= cur_tp2 and not cur_tp2_hit:
                cur_tp2_hit = True
                if use_trailing:
                    cur_trail = cur_tp1
            if not np.isnan(cur_tp3) and low_v[i] <= cur_tp3 and not cur_tp3_hit:
                cur_tp3_hit = True
                if use_trailing:
                    cur_trail = cur_tp2
            if not np.isnan(pre_trail) and high_v[i] >= pre_trail:
                cur_sl_hit = True

        direction[i] = cur_dir
        entry[i] = cur_entry
        sl[i] = cur_sl
        tp1[i] = cur_tp1
        tp2[i] = cur_tp2
        tp3[i] = cur_tp3
        trail[i] = cur_trail
        tp1_hit[i] = cur_tp1_hit
        tp2_hit[i] = cur_tp2_hit
        tp3_hit[i] = cur_tp3_hit
        sl_hit[i] = cur_sl_hit

    return direction, entry, sl, tp1, tp2, tp3, trail, tp1_hit, tp2_hit, tp3_hit, sl_hit


def precision_confluence(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    preset: str = "default",
    ema_fast: int = 9,
    ema_slow: int = 21,
    ema_trend: int = 55,
    rsi_length: int = 13,
    atr_length: int = 14,
    min_score: int = 5,
    sl_mult: float = 1.5,
    tp1_mult: float = 1.0,
    tp2_mult: float = 2.0,
    tp3_mult: float = 3.0,
    use_trailing: bool = True,
    use_structure_sl: bool = True,
    swing_lookback: int = 10,
    htf_step: int = 5,
) -> pd.DataFrame:
    """Preset-aware confluence trend signal engine with risk ladder outputs.

    Inspired by TradingView file 4, implemented as a native v1indicators API.
    """
    if swing_lookback <= 0:
        raise ValueError("swing_lookback must be > 0")
    if tp1_mult <= 0 or tp2_mult <= 0 or tp3_mult <= 0:
        raise ValueError("tp multipliers must be > 0")
    if htf_step <= 0:
        raise ValueError("htf_step must be > 0")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    ef, es, et, rl, al, score_req, sl_eff, resolved_preset = _resolve_profile(
        preset,
        ema_fast,
        ema_slow,
        ema_trend,
        rsi_length,
        atr_length,
        min_score,
        sl_mult,
    )

    ema_fast_s = ema(close_s, ef)
    ema_slow_s = ema(close_s, es)
    ema_trend_s = ema(close_s, et)
    atr_s = atr(high_s, low_s, close_s, length=al)
    rsi_s = rsi(close_s, length=rl)
    macd_df = macd(close_s)
    adx_df = adx(high_s, low_s, close_s, length=14)
    vol_sma = sma(volume_s, 20)
    vwap_s = vwap(high_s, low_s, close_s, volume_s)

    reduced_close, groups = _resample_last(close_s, htf_step)
    htf_fast = ema(reduced_close, ef)
    htf_slow = ema(reduced_close, es)
    htf_fast_full = _expand_group_series(htf_fast, groups, close_s.index, name="PC_HTF_FAST")
    htf_slow_full = _expand_group_series(htf_slow, groups, close_s.index, name="PC_HTF_SLOW")
    htf_bias = pd.Series(
        np.where(htf_fast_full > htf_slow_full, 1, np.where(htf_fast_full < htf_slow_full, -1, 0)),
        index=close_s.index,
        dtype=np.int8,
    )

    adx_col = "ADX_14"
    dmp_col = "DMP_14"
    dmn_col = "DMN_14"

    bull_score = (
        (ema_fast_s > ema_slow_s).astype(np.float64)
        + (close_s > ema_trend_s).astype(np.float64)
        + ((rsi_s > 50.0) & (rsi_s < 75.0)).astype(np.float64)
        + (macd_df["MACD_HIST"] > 0.0).astype(np.float64)
        + (macd_df["MACD"] > macd_df["MACD_SIGNAL"]).astype(np.float64)
        + (close_s > vwap_s).astype(np.float64)
        + (volume_s > vol_sma).astype(np.float64)
        + ((adx_df[adx_col] > 20.0) & (adx_df[dmp_col] > adx_df[dmn_col])).astype(np.float64)
        + (htf_bias == 1).astype(np.float64) * 1.5
        + (close_s > ema_fast_s).astype(np.float64) * 0.5
    )

    bear_score = (
        (ema_fast_s < ema_slow_s).astype(np.float64)
        + (close_s < ema_trend_s).astype(np.float64)
        + ((rsi_s < 50.0) & (rsi_s > 25.0)).astype(np.float64)
        + (macd_df["MACD_HIST"] < 0.0).astype(np.float64)
        + (macd_df["MACD"] < macd_df["MACD_SIGNAL"]).astype(np.float64)
        + (close_s < vwap_s).astype(np.float64)
        + (volume_s > vol_sma).astype(np.float64)
        + ((adx_df[adx_col] > 20.0) & (adx_df[dmn_col] > adx_df[dmp_col])).astype(np.float64)
        + (htf_bias == -1).astype(np.float64) * 1.5
        + (close_s < ema_fast_s).astype(np.float64) * 0.5
    )

    ema_bull_cross = (ema_fast_s > ema_slow_s) & (ema_fast_s.shift(1) <= ema_slow_s.shift(1))
    ema_bear_cross = (ema_fast_s < ema_slow_s) & (ema_fast_s.shift(1) >= ema_slow_s.shift(1))

    bull_momentum = (close_s > ema_fast_s) & (close_s > ema_slow_s)
    bear_momentum = (close_s < ema_fast_s) & (close_s < ema_slow_s)

    raw_buy = ema_bull_cross & bull_momentum & (rsi_s < 75.0) & (bull_score >= float(score_req))
    raw_sell = ema_bear_cross & bear_momentum & (rsi_s > 25.0) & (bear_score >= float(score_req))

    warmup = max(et, 50)
    buy = np.zeros(len(close_s), dtype=bool)
    sell = np.zeros(len(close_s), dtype=bool)
    last_dir = 0
    for i in range(len(close_s)):
        cb = bool(raw_buy.iloc[i]) and last_dir != 1 and i >= warmup
        cs = bool(raw_sell.iloc[i]) and last_dir != -1 and i >= warmup
        if cb and cs:
            cs = False
        if cb:
            last_dir = 1
        elif cs:
            last_dir = -1
        buy[i] = cb
        sell[i] = cs

    swing_low = low_s.rolling(swing_lookback, min_periods=1).min()
    swing_high = high_s.rolling(swing_lookback, min_periods=1).max()

    (
        direction,
        entry,
        sl,
        tp1,
        tp2,
        tp3,
        trail,
        tp1_hit,
        tp2_hit,
        tp3_hit,
        sl_hit,
    ) = _precision_trade_kernel(
        close_s.to_numpy(dtype=np.float64),
        high_s.to_numpy(dtype=np.float64),
        low_s.to_numpy(dtype=np.float64),
        buy,
        sell,
        atr_s.to_numpy(dtype=np.float64),
        float(sl_eff),
        float(tp1_mult),
        float(tp2_mult),
        float(tp3_mult),
        bool(use_structure_sl),
        bool(use_trailing),
        swing_low.to_numpy(dtype=np.float64),
        swing_high.to_numpy(dtype=np.float64),
    )

    return pd.DataFrame(
        {
            "PC_EMA_FAST": ema_fast_s,
            "PC_EMA_SLOW": ema_slow_s,
            "PC_EMA_TREND": ema_trend_s,
            "PC_ATR": atr_s,
            "PC_RSI": rsi_s,
            "PC_MACD": macd_df["MACD"],
            "PC_MACD_SIGNAL": macd_df["MACD_SIGNAL"],
            "PC_MACD_HIST": macd_df["MACD_HIST"],
            "PC_ADX": adx_df[adx_col],
            "PC_DI_PLUS": adx_df[dmp_col],
            "PC_DI_MINUS": adx_df[dmn_col],
            "PC_VWAP": vwap_s,
            "PC_HTF_BIAS": htf_bias,
            "PC_BULL_SCORE": bull_score,
            "PC_BEAR_SCORE": bear_score,
            "PC_BUY": pd.Series(buy, index=close_s.index),
            "PC_SELL": pd.Series(sell, index=close_s.index),
            "PC_DIR": pd.Series(direction, index=close_s.index),
            "PC_ENTRY": pd.Series(entry, index=close_s.index),
            "PC_SL": pd.Series(sl, index=close_s.index),
            "PC_TP1": pd.Series(tp1, index=close_s.index),
            "PC_TP2": pd.Series(tp2, index=close_s.index),
            "PC_TP3": pd.Series(tp3, index=close_s.index),
            "PC_TRAIL": pd.Series(trail, index=close_s.index),
            "PC_TP1_HIT": pd.Series(tp1_hit, index=close_s.index),
            "PC_TP2_HIT": pd.Series(tp2_hit, index=close_s.index),
            "PC_TP3_HIT": pd.Series(tp3_hit, index=close_s.index),
            "PC_SL_HIT": pd.Series(sl_hit, index=close_s.index),
            "PC_PROFILE": pd.Series(np.repeat(resolved_preset, len(close_s)), index=close_s.index),
        },
        index=close_s.index,
    )
