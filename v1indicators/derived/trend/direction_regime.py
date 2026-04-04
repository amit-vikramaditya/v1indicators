import numpy as np
import pandas as pd

from .._utils import check_series
from ...foundational.overlap.ema import ema
from ...foundational.overlap.hma import hma
from ...foundational.overlap.sma import sma
from ...foundational.overlap.wma import wma


def _selected_ma(close: pd.Series, length: int, ma_type: str) -> pd.Series:
    mt = ma_type.lower()
    if mt == "ema":
        return ema(close, length)
    if mt == "sma":
        return sma(close, length)
    if mt == "wma":
        out = wma(close, length)
        out.name = f"WMA_{length}"
        return out
    if mt == "hma":
        return hma(close, length)
    raise ValueError("ma_type must be one of ['ema', 'sma', 'wma', 'hma']")


def direction_regime(
    close: pd.Series,
    ma_length: int = 50,
    ma_type: str = "ema",
    threshold_percent: float = 0.3,
) -> pd.DataFrame:
    """
    Regime classification: bullish, bearish, or sideways.

    A regime is identified by price distance from a selected moving average,
    with an adaptive percentage threshold.
    """
    if ma_length <= 0:
        raise ValueError("ma_length must be > 0")
    if threshold_percent < 0:
        raise ValueError("threshold_percent must be >= 0")

    close_s = check_series(close, "close")
    ma = _selected_ma(close_s, ma_length, ma_type)

    threshold = close_s * (threshold_percent * 0.01)
    price_change = (close_s - ma).abs()

    uptrend = (close_s > ma) & (price_change > threshold)
    downtrend = (close_s < ma) & (price_change > threshold)
    sideways = ~(uptrend | downtrend)

    trend = pd.Series(
        np.where(uptrend, 1, np.where(downtrend, -1, 0)),
        index=close_s.index,
        dtype=np.int8,
    )

    trend_change = trend != trend.shift(1)

    return pd.DataFrame(
        {
            "DIRECTION_MA": ma,
            "UPTREND": uptrend,
            "DOWNTREND": downtrend,
            "SIDEWAYS": sideways,
            "TREND": trend,
            "TREND_CHANGE": trend_change.fillna(False),
        }
    )
