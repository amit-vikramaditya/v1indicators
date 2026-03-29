import pandas as pd

from .._utils import check_series


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 9,
    signal: int = 3,
) -> pd.DataFrame:
    """KDJ oscillator."""
    if length <= 0 or signal <= 0:
        raise ValueError("length and signal must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    ll = low_s.rolling(length).min()
    hh = high_s.rolling(length).max()
    rsv = 100.0 * (close_s - ll) / (hh - ll).replace(0.0, pd.NA)
    k = rsv.ewm(span=signal, adjust=False).mean()
    d = k.ewm(span=signal, adjust=False).mean()
    j = 3.0 * k - 2.0 * d

    return pd.DataFrame({"KDJ_K": k, "KDJ_D": d, "KDJ_J": j})
