import numpy as np
import pandas as pd

from .._utils import check_series
from ...foundational.volatility.atr import atr


def swing_leg_profile(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    swing_length: int = 50,
    atr_length: int = 200,
    bin_atr_mult: float = 0.5,
    min_bin_count: int = 5,
) -> pd.DataFrame:
    """Swing-leg volume profile summary metrics.

    Inspired by TradingView file 3, converted to non-visual per-bar profile outputs.
    """
    if swing_length <= 1:
        raise ValueError("swing_length must be > 1")
    if atr_length <= 0:
        raise ValueError("atr_length must be > 0")
    if bin_atr_mult <= 0:
        raise ValueError("bin_atr_mult must be > 0")
    if min_bin_count <= 0:
        raise ValueError("min_bin_count must be > 0")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    n = len(close_s)
    highest = high_s.rolling(swing_length, min_periods=swing_length).max()
    lowest = low_s.rolling(swing_length, min_periods=swing_length).min()
    atr_s = atr(high_s, low_s, close_s, length=atr_length)

    dir_v = np.zeros(n, dtype=np.int8)
    swing_high_v = np.full(n, np.nan, dtype=np.float64)
    swing_low_v = np.full(n, np.nan, dtype=np.float64)

    poc_v = np.full(n, np.nan, dtype=np.float64)
    total_v = np.full(n, np.nan, dtype=np.float64)
    buy_v = np.full(n, np.nan, dtype=np.float64)
    sell_v = np.full(n, np.nan, dtype=np.float64)
    delta_v = np.full(n, np.nan, dtype=np.float64)
    bins_v = np.full(n, np.nan, dtype=np.float64)

    leg_start_v = np.full(n, -1, dtype=np.int64)
    leg_end_v = np.full(n, -1, dtype=np.int64)

    is_down_move = False
    prev_is_down_move = False

    high_idx = -1
    low_idx = -1
    high_price = np.nan
    low_price = np.nan

    cur_poc = np.nan
    cur_total = np.nan
    cur_buy = np.nan
    cur_sell = np.nan
    cur_delta = np.nan
    cur_bins = np.nan
    cur_leg_start = -1
    cur_leg_end = -1

    open_v = open_s.to_numpy(dtype=np.float64)
    high_v = high_s.to_numpy(dtype=np.float64)
    low_v = low_s.to_numpy(dtype=np.float64)
    close_v = close_s.to_numpy(dtype=np.float64)
    volume_arr = volume_s.to_numpy(dtype=np.float64)
    highest_v = highest.to_numpy(dtype=np.float64)
    lowest_v = lowest.to_numpy(dtype=np.float64)
    atr_v = atr_s.to_numpy(dtype=np.float64)

    for i in range(n):
        if not np.isnan(highest_v[i]) and high_v[i] >= highest_v[i]:
            is_down_move = True
        if not np.isnan(lowest_v[i]) and low_v[i] <= lowest_v[i]:
            is_down_move = False

        if i >= 1:
            if not np.isnan(highest_v[i - 1]) and high_v[i - 1] == highest_v[i - 1] and high_v[i] < highest_v[i]:
                high_idx = i - 1
                high_price = high_v[i - 1]
            if not np.isnan(lowest_v[i - 1]) and low_v[i - 1] == lowest_v[i - 1] and low_v[i] > lowest_v[i]:
                low_idx = i - 1
                low_price = low_v[i - 1]

        if i > 0 and is_down_move != prev_is_down_move and high_idx >= 0 and low_idx >= 0:
            leg_start = min(high_idx, low_idx)
            leg_end = max(high_idx, low_idx)

            if leg_end > leg_start and not np.isnan(high_price) and not np.isnan(low_price):
                swing_bottom = min(high_price, low_price)
                swing_top = max(high_price, low_price)
                swing_range = swing_top - swing_bottom

                atr_i = atr_v[i]
                bin_size = atr_i * bin_atr_mult
                if np.isnan(bin_size) or bin_size <= 0.0:
                    bin_size = swing_range / float(max(min_bin_count, 1)) if swing_range > 0 else 1e-9

                bin_count = int(swing_range / bin_size) if bin_size > 0 else min_bin_count
                if bin_count < min_bin_count:
                    bin_count = min_bin_count
                if bin_count <= 0:
                    bin_count = 1

                step = swing_range / float(bin_count) if swing_range > 0 else max(bin_size, 1e-9)
                if step <= 0:
                    step = 1e-9

                vol_bins = np.zeros(bin_count, dtype=np.float64)
                buy_bins = np.zeros(bin_count, dtype=np.float64)
                sell_bins = np.zeros(bin_count, dtype=np.float64)

                for j in range(leg_start, leg_end + 1):
                    cj = close_v[j]
                    oj = open_v[j]
                    vj = volume_arr[j]
                    if np.isnan(cj) or np.isnan(oj) or np.isnan(vj):
                        continue

                    idx = int((cj - swing_bottom) / step)
                    if idx < 0:
                        idx = 0
                    elif idx >= bin_count:
                        idx = bin_count - 1

                    vol_bins[idx] += vj
                    if cj > oj:
                        buy_bins[idx] += vj
                    else:
                        sell_bins[idx] += vj

                total = vol_bins.sum()
                if total > 0.0:
                    poc_idx = int(np.argmax(vol_bins))
                    cur_poc = swing_bottom + (poc_idx + 0.5) * step
                    cur_total = total
                    cur_buy = buy_bins.sum()
                    cur_sell = sell_bins.sum()
                    cur_delta = ((cur_buy - cur_sell) / cur_total) * 100.0
                    cur_bins = float(bin_count)
                    cur_leg_start = leg_start
                    cur_leg_end = leg_end

        prev_is_down_move = is_down_move

        dir_v[i] = -1 if is_down_move else 1
        swing_high_v[i] = high_price
        swing_low_v[i] = low_price
        poc_v[i] = cur_poc
        total_v[i] = cur_total
        buy_v[i] = cur_buy
        sell_v[i] = cur_sell
        delta_v[i] = cur_delta
        bins_v[i] = cur_bins
        leg_start_v[i] = cur_leg_start
        leg_end_v[i] = cur_leg_end

    return pd.DataFrame(
        {
            "SLP_DIR": pd.Series(dir_v, index=close_s.index),
            "SLP_SWING_HIGH": pd.Series(swing_high_v, index=close_s.index),
            "SLP_SWING_LOW": pd.Series(swing_low_v, index=close_s.index),
            "SLP_POC": pd.Series(poc_v, index=close_s.index),
            "SLP_TOTAL_VOL": pd.Series(total_v, index=close_s.index),
            "SLP_BUY_VOL": pd.Series(buy_v, index=close_s.index),
            "SLP_SELL_VOL": pd.Series(sell_v, index=close_s.index),
            "SLP_DELTA_PCT": pd.Series(delta_v, index=close_s.index),
            "SLP_BIN_COUNT": pd.Series(bins_v, index=close_s.index),
            "SLP_LEG_START": pd.Series(leg_start_v, index=close_s.index),
            "SLP_LEG_END": pd.Series(leg_end_v, index=close_s.index),
        },
        index=close_s.index,
    )
