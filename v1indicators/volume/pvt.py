import pandas as pd

from .._utils import check_series


def pvt(close: pd.Series, volume: pd.Series, drift: int = 1) -> pd.Series:
    """Price-Volume Trend (PVT)."""
    if drift <= 0:
        raise ValueError("drift must be > 0")

    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    out = (close_s.pct_change(drift).fillna(0.0) * volume_s).cumsum()
    out.name = "PVT"
    return out
