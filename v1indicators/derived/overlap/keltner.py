import pandas as pd
from .._utils import check_series
from ...foundational.overlap.ema import ema
from ...foundational.volatility.atr import atr

def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    atr_length: int = 10,
    mult: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels (EMA ± ATR bands)."""

    if min(length, atr_length) <= 0:
        raise ValueError("lengths must be > 0")

    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")

    mid = ema(close, length)
    atr_v = atr(high, low, close, atr_length)

    upper = mid + mult * atr_v
    lower = mid - mult * atr_v

    return pd.DataFrame({
        "KELTNER_MID": mid,
        "KELTNER_UPPER": upper,
        "KELTNER_LOWER": lower,
    })
