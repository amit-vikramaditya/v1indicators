import pandas as pd

from .._utils import check_series


def kst(
    close: pd.Series,
    r1: int = 10,
    r2: int = 15,
    r3: int = 20,
    r4: int = 30,
    n1: int = 10,
    n2: int = 10,
    n3: int = 10,
    n4: int = 15,
    signal: int = 9,
) -> pd.DataFrame:
    """Know Sure Thing oscillator and signal."""
    if min(r1, r2, r3, r4, n1, n2, n3, n4, signal) <= 0:
        raise ValueError("all periods must be > 0")

    close_s = check_series(close, "close")

    rcma1 = (100.0 * close_s.pct_change(r1)).rolling(n1).mean()
    rcma2 = (100.0 * close_s.pct_change(r2)).rolling(n2).mean()
    rcma3 = (100.0 * close_s.pct_change(r3)).rolling(n3).mean()
    rcma4 = (100.0 * close_s.pct_change(r4)).rolling(n4).mean()

    k = rcma1 + 2.0 * rcma2 + 3.0 * rcma3 + 4.0 * rcma4
    s = k.rolling(signal).mean()
    return pd.DataFrame({"KST": k, "KST_SIGNAL": s})
