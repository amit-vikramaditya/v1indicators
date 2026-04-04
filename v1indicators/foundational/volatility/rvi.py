import numpy as np
import pandas as pd

from ..._utils import check_series


def rvi(close: pd.Series, length: int = 14, scalar: float = 100.0, drift: int = 1) -> pd.Series:
    """Relative Volatility Index using close series."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if drift <= 0:
        raise ValueError("drift must be > 0")

    close_s = check_series(close, "close")
    std = close_s.rolling(length).std()

    delta = close_s.diff(drift)
    pos = (delta > 0.0).astype(float)
    neg = (delta < 0.0).astype(float)

    pos_avg = (pos * std).ewm(span=length, adjust=False).mean()
    neg_avg = (neg * std).ewm(span=length, adjust=False).mean()

    out = scalar * pos_avg / (pos_avg + neg_avg).replace(0.0, np.nan)
    out.name = f"RVI_{length}"
    return out
