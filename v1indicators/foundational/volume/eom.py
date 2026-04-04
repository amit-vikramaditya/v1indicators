import numpy as np
import pandas as pd

from .._utils import check_series


def eom(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
    divisor: float = 100_000_000.0,
    drift: int = 1,
) -> pd.Series:
    """Ease of Movement."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if drift <= 0:
        raise ValueError("drift must be > 0")
    if divisor <= 0:
        raise ValueError("divisor must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    _ = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    hl2 = 0.5 * (high_s + low_s)
    distance = hl2 - hl2.shift(drift)
    box_ratio = (volume_s / divisor) / (high_s - low_s).replace(0.0, np.nan)

    raw = distance / box_ratio
    out = raw.rolling(length).mean()
    out.name = f"EOM_{length}_{int(divisor)}"
    return out
