import pandas as pd

from .._utils import check_series
from ..volatility.atr import atr


def aberration(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 5,
    atr_length: int = 15,
) -> pd.DataFrame:
    """Aberration channels based on typical price and ATR."""
    if length <= 0 or atr_length <= 0:
        raise ValueError("length and atr_length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    zg = ((high_s + low_s + close_s) / 3.0).rolling(length).mean()
    atr_s = atr(high_s, low_s, close_s, length=atr_length)

    return pd.DataFrame({"ABER_ZG": zg, "ABER_SG": zg + atr_s, "ABER_XG": zg - atr_s})
