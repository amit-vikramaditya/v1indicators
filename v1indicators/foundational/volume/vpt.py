import pandas as pd

from .._utils import check_series


def vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume Price Trend (VPT).

    VPT = cumulative( volume * pct_change(close) )
    """
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    result = (volume_s * close_s.pct_change().fillna(0.0)).cumsum()
    result.name = "VPT"
    return result
