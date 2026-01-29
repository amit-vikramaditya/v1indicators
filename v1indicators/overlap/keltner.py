import pandas as pd
from ..overlap.ema import ema
from ..volatility.atr import atr


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    atr_length: int = 10,
    mult: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels (EMA ± ATR bands)."""

    if not all(isinstance(x, pd.Series) for x in (high, low, close)):
        raise TypeError("high, low, close must be pandas Series")

    if min(length, atr_length) <= 0:
        raise ValueError("lengths must be > 0")

    mid = ema(close, length)
    atr_v = atr(high, low, close, atr_length)

    upper = mid + mult * atr_v
    lower = mid - mult * atr_v

    return pd.DataFrame({
        "keltner_mid": mid,
        "keltner_upper": upper,
        "keltner_lower": lower,
    })

