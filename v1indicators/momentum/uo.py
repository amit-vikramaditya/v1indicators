import pandas as pd

from .._utils import check_series
from .ultimate_oscillator import ultimate_oscillator


def uo(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short: int = 7,
    medium: int = 14,
    long: int = 28,
) -> pd.Series:
    """Alias for Ultimate Oscillator (UO)."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    return ultimate_oscillator(high_s, low_s, close_s, short=short, medium=medium, long=long)
