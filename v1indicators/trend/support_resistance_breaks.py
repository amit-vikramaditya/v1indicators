import numpy as np
import pandas as pd

from .._utils import check_series


def support_resistance_breaks(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    left: int = 15,
    right: int = 15,
    volume_fast: int = 5,
    volume_slow: int = 10,
    volume_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    Pivot support/resistance with breakout classification and volume filter.

    Breakouts are filtered by volume oscillator and split into body-dominant
    and wick-dominant breaks.
    """
    if left <= 0 or right <= 0:
        raise ValueError("left and right must be > 0")
    if volume_fast <= 0 or volume_slow <= 0:
        raise ValueError("volume_fast and volume_slow must be > 0")

    open_s = check_series(open_, "open_")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    window = left + right + 1
    pivot_high = high_s.where(high_s == high_s.rolling(window).max().shift(-right))
    pivot_low = low_s.where(low_s == low_s.rolling(window).min().shift(-right))

    resistance = pivot_high.ffill()
    support = pivot_low.ffill()

    vol_fast = volume_s.ewm(span=volume_fast, adjust=False).mean()
    vol_slow = volume_s.ewm(span=volume_slow, adjust=False).mean().replace(0.0, np.nan)
    vol_osc = 100.0 * (vol_fast - vol_slow) / vol_slow

    prev_close = close_s.shift(1)
    prev_res = resistance.shift(1)
    prev_sup = support.shift(1)

    cross_up = (prev_close <= prev_res) & (close_s > prev_res)
    cross_down = (prev_close >= prev_sup) & (close_s < prev_sup)

    bullish_wick_shape = (open_s - low_s) > (close_s - open_s)
    bearish_wick_shape = (open_s - close_s) < (high_s - open_s)

    high_volume = vol_osc > volume_threshold

    break_resistance = cross_up & high_volume & (~bullish_wick_shape)
    break_support = cross_down & high_volume & (~bearish_wick_shape)
    bull_wick_break = cross_up & high_volume & bullish_wick_shape
    bear_wick_break = cross_down & high_volume & bearish_wick_shape

    return pd.DataFrame(
        {
            "PIVOT_HIGH": pivot_high,
            "PIVOT_LOW": pivot_low,
            "RESISTANCE": resistance,
            "SUPPORT": support,
            "VOLUME_OSC": vol_osc,
            "BREAK_RESISTANCE": break_resistance.fillna(False),
            "BREAK_SUPPORT": break_support.fillna(False),
            "BULL_WICK_BREAK": bull_wick_break.fillna(False),
            "BEAR_WICK_BREAK": bear_wick_break.fillna(False),
        }
    )
