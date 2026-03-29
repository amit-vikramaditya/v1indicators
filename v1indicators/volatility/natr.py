import numpy as np
import pandas as pd

from .._utils import check_series
from .atr import atr


def natr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Normalized ATR: 100 * ATR / close."""
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close").replace(0.0, np.nan)

    atr_s = atr(high_s, low_s, close_s, length=length, mamode="ema", drift=1)
    out = 100.0 * atr_s / close_s
    out.name = f"NATR_{length}"
    return out
