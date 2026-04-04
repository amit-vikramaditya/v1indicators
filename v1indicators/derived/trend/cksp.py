import pandas as pd

from .._utils import check_series
from ...foundational.volatility.atr import atr


def cksp(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    p: int = 10,
    x: float = 1.0,
    q: int = 9,
) -> pd.DataFrame:
    """Chande Kroll Stop (long/short stops)."""
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be > 0")
    if x <= 0:
        raise ValueError("x must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    atr_s = atr(high_s, low_s, close_s, length=p)
    long_stop = (high_s.rolling(p).max() - x * atr_s).rolling(q).max()
    short_stop = (low_s.rolling(p).min() + x * atr_s).rolling(q).min()

    return pd.DataFrame({f"CKSPL_{p}_{x}_{q}": long_stop, f"CKSPS_{p}_{x}_{q}": short_stop})
