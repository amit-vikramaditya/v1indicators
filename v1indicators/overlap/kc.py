import pandas as pd

from .._utils import check_series
from .keltner import keltner


def kc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    atr_length: int = 10,
    mult: float = 2.0,
) -> pd.DataFrame:
    """Alias for Keltner Channel (KC)."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    return keltner(high_s, low_s, close_s, length=length, atr_length=atr_length, mult=mult)
