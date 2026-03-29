import pandas as pd

from .._utils import check_series


def pvol(close: pd.Series, volume: pd.Series, signed: bool = False) -> pd.Series:
    """Price-Volume product."""
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    out = close_s * volume_s
    if signed:
        out = out * close_s.diff().fillna(0.0).apply(lambda x: 1.0 if x >= 0.0 else -1.0)

    out.name = "PVOL"
    return out
