import pandas as pd
import numpy as np
from .._utils import check_series

def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price, cumulative from the first bar of the input.

    ANCHOR SEMANTICS: this is an anchored (cumulative) VWAP. There is no
    automatic daily/session reset — the anchor is wherever the supplied
    series starts, so prepending history changes every value. For a
    session VWAP, slice the input to the session before calling.
    Strictly causal: bar i uses bars <= i only.
    """

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

