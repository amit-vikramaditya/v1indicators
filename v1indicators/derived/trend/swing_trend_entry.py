import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series
from ...foundational.overlap.ema import ema
from ...foundational.overlap.hma import hma
from ...foundational.overlap.sma import sma
from ...foundational.overlap.wma import wma


@njit
def _dynamic_gap_source(close_v: np.ndarray, gap_fraction: float) -> np.ndarray:
    n = close_v.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        gap = int(i * gap_fraction)
        src_idx = i - gap
        if src_idx >= 0:
            out[i] = close_v[src_idx]

    return out


def _selected_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    mt = ma_type.lower()
    if mt == "ema":
        return ema(series, length)
    if mt == "sma":
        return sma(series, length)
    if mt == "wma":
        out = wma(series, length)
        out.name = f"WMA_{length}"
        return out
    if mt == "hma":
        return hma(series, length)
    raise ValueError("ma_type must be one of ['ema', 'sma', 'wma', 'hma']")


def swing_trend_entry(
    close: pd.Series,
    ma_length: int = 200,
    long_ma_length: int = 250,
    time_gap_percent: float = 0.4,
    threshold_percent: float = 0.3,
    ma_type: str = "ema",
) -> pd.DataFrame:
    """
    Swing trend entry helper from adaptive MA regime logic.

    Builds a dynamic-gap source, computes short/long moving averages, and
    emits trend-direction and MA-touch alert flags.
    """
    if ma_length <= 0 or long_ma_length <= 0:
        raise ValueError("ma_length and long_ma_length must be > 0")
    if time_gap_percent < 0:
        raise ValueError("time_gap_percent must be >= 0")
    if threshold_percent < 0:
        raise ValueError("threshold_percent must be >= 0")

    close_s = check_series(close, "close")

    gap_fraction = time_gap_percent / 100.0
    gap_source_v = _dynamic_gap_source(close_s.to_numpy(dtype=np.float64), float(gap_fraction))
    gap_source = pd.Series(gap_source_v, index=close_s.index, name="GAP_SOURCE")

    ma = _selected_ma(gap_source, ma_length, ma_type)
    long_ma = _selected_ma(gap_source, long_ma_length, ma_type)

    ma_diff = ma - long_ma
    threshold = ma * (threshold_percent * 0.01)

    bullish = (ma_diff > threshold) & (gap_source > ma)
    bearish = (ma_diff < -threshold) & (gap_source < ma)
    sideways = ~(bullish | bearish)

    trend = pd.Series(
        np.where(bullish, 1, np.where(bearish, -1, 0)),
        index=close_s.index,
        dtype=np.int8,
    )

    touch_ma = (
        ((close_s.shift(1) > ma.shift(1)) & (close_s <= ma))
        | ((close_s.shift(1) < ma.shift(1)) & (close_s >= ma))
    )

    trend_change = trend != trend.shift(1)

    return pd.DataFrame(
        {
            "GAP_SOURCE": gap_source,
            "SWING_MA": ma,
            "SWING_LONG_MA": long_ma,
            "BULLISH": bullish,
            "BEARISH": bearish,
            "SIDEWAYS": sideways,
            "TREND": trend,
            "TREND_CHANGE": trend_change.fillna(False),
            "TOUCH_MA": touch_ma.fillna(False),
        }
    )
