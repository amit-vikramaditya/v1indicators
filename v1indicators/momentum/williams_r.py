import numpy as np
import pandas as pd

from .._utils import check_series


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """
    Williams %R oscillator.

    %R = -100 * (HighestHigh(length) - Close) / (HighestHigh(length) - LowestLow(length))
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    highest = high_s.rolling(length).max()
    lowest = low_s.rolling(length).min()
    range_ = (highest - lowest).replace(0.0, np.nan)

    result = -100.0 * (highest - close_s) / range_
    result.name = f"WILLR_{length}"
    return result
