import pandas as pd

from .._utils import check_series
from .williams_r import williams_r


def willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Alias for Williams %R."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    return williams_r(high_s, low_s, close_s, length=length)
