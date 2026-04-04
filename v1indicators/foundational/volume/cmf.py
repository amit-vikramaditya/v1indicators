import numpy as np
import pandas as pd

from .._utils import check_series


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 20,
) -> pd.Series:
    """
    Chaikin Money Flow (CMF).

    CMF = SUM(MFM * Volume, length) / SUM(Volume, length)
    MFM = ((Close - Low) - (High - Close)) / (High - Low)
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    hl_range = (high_s - low_s).replace(0.0, np.nan)
    mfm = ((close_s - low_s) - (high_s - close_s)) / hl_range
    mfv = mfm * volume_s

    numerator = mfv.rolling(length).sum()
    denominator = volume_s.rolling(length).sum().replace(0.0, np.nan)
    result = numerator / denominator
    result.name = f"CMF_{length}"
    return result
