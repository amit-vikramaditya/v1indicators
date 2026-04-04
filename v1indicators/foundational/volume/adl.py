import numpy as np
import pandas as pd

from .._utils import check_series


def adl(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Accumulation/Distribution Line (ADL).
    """
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    hl_range = (high_s - low_s).replace(0.0, np.nan)
    mfm = ((close_s - low_s) - (high_s - close_s)) / hl_range
    mfv = mfm.fillna(0.0) * volume_s

    result = mfv.cumsum()
    result.name = "ADL"
    return result
