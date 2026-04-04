import pandas as pd
import numpy as np
from .._utils import check_series

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price (cumulative VWAP)."""

    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")
    volume = check_series(volume, "volume")

    typical = (high + low + close) / 3.0

    cum_vol = volume.cumsum()
    cum_pv = (typical * volume).cumsum()
    
    # Handle zero volume
    vwap_val = cum_pv / cum_vol.replace(0, np.nan)
    vwap_val.name = "VWAP"
    
    return vwap_val

