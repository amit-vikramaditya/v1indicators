import numpy as np
import pandas as pd

from .._utils import check_series
from ...foundational.overlap.ema import ema
from ...foundational.overlap.hma import hma
from ...foundational.overlap.rma import rma
from ...foundational.overlap.sma import sma
from ...foundational.overlap.t3 import t3
from ...foundational.overlap.tema import tema
from ...foundational.overlap.vwma import vwma
from ...foundational.overlap.wma import wma


def _selected_ma(close: pd.Series, volume: pd.Series | None, length: int, ma_type: str, t3_factor: float) -> pd.Series:
    mt = ma_type.lower()
    if mt == "sma":
        return sma(close, length)
    if mt == "ema":
        return ema(close, length)
    if mt == "wma":
        out = wma(close, length)
        out.name = f"WMA_{length}"
        return out
    if mt == "hma":
        return hma(close, length)
    if mt == "vwma":
        if volume is None:
            raise ValueError("volume is required when ma_type is 'vwma'")
        return vwma(close, volume, length)
    if mt == "rma":
        return rma(close, length)
    if mt == "tema":
        return tema(close, length)
    if mt == "t3":
        return t3(close, length=length, factor=t3_factor)
    raise ValueError("ma_type must be one of ['sma','ema','wma','hma','vwma','rma','tema','t3']")


def multi_ma(
    close: pd.Series,
    length1: int = 20,
    ma_type1: str = "ema",
    length2: int = 50,
    ma_type2: str = "ema",
    volume: pd.Series | None = None,
    smoothing: int = 2,
    t3_factor: float = 0.7,
) -> pd.DataFrame:
    """
    Multi-MA engine with crossover and slope-state flags.

    Inspired by TradingView "ultimate MA" style overlays.
    """
    if length1 <= 0 or length2 <= 0:
        raise ValueError("length1 and length2 must be > 0")
    if smoothing <= 0:
        raise ValueError("smoothing must be > 0")
    if t3_factor <= 0:
        raise ValueError("t3_factor must be > 0")

    close_s = check_series(close, "close")
    vol_s = check_series(volume, "volume") if volume is not None else None

    ma1 = _selected_ma(close_s, vol_s, length1, ma_type1, t3_factor)
    ma2 = _selected_ma(close_s, vol_s, length2, ma_type2, t3_factor)

    ma1_up = ma1 >= ma1.shift(smoothing)
    ma1_down = ma1 < ma1.shift(smoothing)

    cross_up = (ma1.shift(1) <= ma2.shift(1)) & (ma1 > ma2)
    cross_down = (ma1.shift(1) >= ma2.shift(1)) & (ma1 < ma2)

    trend = pd.Series(
        np.where(ma1 > ma2, 1, np.where(ma1 < ma2, -1, 0)),
        index=close_s.index,
        dtype=np.int8,
    )

    return pd.DataFrame(
        {
            "MA_FAST": ma1,
            "MA_SLOW": ma2,
            "MA_FAST_UP": ma1_up.fillna(False),
            "MA_FAST_DOWN": ma1_down.fillna(False),
            "MA_CROSS_UP": cross_up.fillna(False),
            "MA_CROSS_DOWN": cross_down.fillna(False),
            "MA_TREND": trend,
        }
    )
