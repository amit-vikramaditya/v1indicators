import numpy as np
import pandas as pd

from .._utils import check_series


def _adl_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    hl = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * volume
    return mfv.cumsum()


def adosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """
    Chaikin Accumulation/Distribution Oscillator.

    ADOSC = EMA(ADL, fast) - EMA(ADL, slow)
    """
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    adl = _adl_line(high_s, low_s, close_s, volume_s)
    out = adl.ewm(span=fast, adjust=False).mean() - adl.ewm(span=slow, adjust=False).mean()
    out.name = f"ADOSC_{fast}_{slow}"
    return out
