import numpy as np
import pandas as pd

from .._utils import check_series


def vwma(close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    """
    Volume Weighted Moving Average (VWMA).

    VWMA = SUM(close * volume, length) / SUM(volume, length)
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    weighted_sum = (close_s * volume_s).rolling(length).sum()
    volume_sum = volume_s.rolling(length).sum().replace(0.0, np.nan)

    result = weighted_sum / volume_sum
    result.name = f"VWMA_{length}"
    return result
