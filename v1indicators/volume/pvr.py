import pandas as pd

from .._utils import check_series


def pvr(close: pd.Series, volume: pd.Series, drift: int = 1) -> pd.Series:
    """Price Volume Rank."""
    if drift <= 0:
        raise ValueError("drift must be > 0")

    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    cd = close_s.diff(drift).fillna(0.0)
    vd = volume_s.diff(drift).fillna(0.0)

    out = pd.Series(index=close_s.index, dtype=float)
    out[(cd >= 0.0) & (vd >= 0.0)] = 1.0
    out[(cd >= 0.0) & (vd < 0.0)] = 2.0
    out[(cd < 0.0) & (vd >= 0.0)] = 3.0
    out[(cd < 0.0) & (vd < 0.0)] = 4.0
    out.name = "PVR"
    return out
