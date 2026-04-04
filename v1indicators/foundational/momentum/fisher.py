import numpy as np
import pandas as pd

from .._utils import check_series


def fisher(high: pd.Series, low: pd.Series, length: int = 9, signal: int = 1) -> pd.DataFrame:
    """Fisher Transform and signal."""
    if length <= 1 or signal <= 0:
        raise ValueError("length must be > 1 and signal must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    hl2 = (high_s + low_s) / 2.0

    nmin = hl2.rolling(length).min()
    nmax = hl2.rolling(length).max()
    x = 2.0 * ((hl2 - nmin) / (nmax - nmin).replace(0.0, np.nan) - 0.5)
    x = x.clip(-0.999, 0.999)
    fish = 0.5 * np.log((1.0 + x) / (1.0 - x))
    sig = fish.shift(signal)

    return pd.DataFrame({f"FISHERT_{length}_{signal}": fish, f"FISHERTs_{length}_{signal}": sig})
