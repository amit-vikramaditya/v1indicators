import numpy as np
import pandas as pd

from ..._utils import check_series


def chop(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Choppiness Index."""
    if length <= 1:
        raise ValueError("length must be > 1")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    prev_close = close_s.shift(1)
    tr = pd.concat(
        [high_s - low_s, (high_s - prev_close).abs(), (low_s - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr_sum = tr.rolling(length).sum()
    hh = high_s.rolling(length).max()
    ll = low_s.rolling(length).min()
    rng = (hh - ll).replace(0.0, np.nan)

    out = 100.0 * (np.log10(atr_sum / rng) / np.log10(float(length)))
    out.name = f"CHOP_{length}"
    return out
