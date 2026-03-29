import pandas as pd

from .._utils import check_series
from .rsi import rsi


def rsx(close: pd.Series, length: int = 14) -> pd.Series:
    """RSX approximation via double-smoothed RSI."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    r = rsi(close_s, length=length)
    out = r.ewm(span=length, adjust=False).mean().ewm(span=length, adjust=False).mean()
    out.name = f"RSX_{length}"
    return out
