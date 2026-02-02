import pandas as pd
import numpy as np
from .._utils import check_series

def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    smooth: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D)."""
    
    if min(length, smooth) <= 0:
        raise ValueError("length and smooth must be > 0")

    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")

    lowest = low.rolling(length).min()
    highest = high.rolling(length).max()

    range_ = highest - lowest
    
    # Avoid div by zero using replace
    k = 100 * (close - lowest) / range_.replace(0, np.nan)
    d = k.rolling(smooth).mean()

    return pd.DataFrame({
        "STOCH_K": k,
        "STOCH_D": d,
    })

