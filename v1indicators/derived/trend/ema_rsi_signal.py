import numpy as np
import pandas as pd

from .._utils import check_series
from ...foundational.momentum.rsi import rsi
from ...foundational.overlap.ema import ema


def ema_rsi_signal(
    close: pd.Series,
    fast_length: int = 13,
    slow_length: int = 41,
    rsi_length: int = 14,
    rsi_buy_level: float = 63.0,
    rsi_sell_level: float = 27.0,
) -> pd.DataFrame:
    """
    EMA + RSI composite signal.

    Produces long/short setup and simple EMA-based exit flags.
    """
    if min(fast_length, slow_length, rsi_length) <= 0:
        raise ValueError("fast_length, slow_length, and rsi_length must be > 0")

    close_s = check_series(close, "close")

    fast_ema = ema(close_s, fast_length)
    slow_ema = ema(close_s, slow_length)
    rsi_line = rsi(close_s, rsi_length)

    long_signal = (fast_ema > slow_ema) & (rsi_line > rsi_buy_level)
    short_signal = (fast_ema < slow_ema) & (rsi_line < rsi_sell_level)

    exit_long = close_s < slow_ema
    exit_short = close_s > slow_ema

    trend = pd.Series(
        np.where(fast_ema > slow_ema, 1, np.where(fast_ema < slow_ema, -1, 0)),
        index=close_s.index,
        dtype=np.int8,
    )

    return pd.DataFrame(
        {
            "EMA_FAST": fast_ema,
            "EMA_SLOW": slow_ema,
            "RSI": rsi_line,
            "LONG_SIGNAL": long_signal,
            "SHORT_SIGNAL": short_signal,
            "EXIT_LONG": exit_long,
            "EXIT_SHORT": exit_short,
            "TREND": trend,
        }
    )
