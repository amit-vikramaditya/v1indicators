import pandas as pd

from .._utils import check_series


def tma(close: pd.Series, length: int = 20) -> pd.Series:
    """
    Triangular Moving Average (TMA).

    TMA is a double-smoothed SMA with staggered windows.
    For odd n: SMA(SMA(close, (n+1)/2), (n+1)/2)
    For even n: SMA(SMA(close, n/2), n/2 + 1)
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")

    if length % 2 == 1:
        w1 = (length + 1) // 2
        w2 = w1
    else:
        w1 = length // 2
        w2 = w1 + 1

    out = close_s.rolling(w1).mean().rolling(w2).mean()
    out.name = f"TMA_{length}"
    return out
