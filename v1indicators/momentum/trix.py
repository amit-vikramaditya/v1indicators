import pandas as pd

from .._utils import check_series


def trix(close: pd.Series, length: int = 15, drift: int = 1) -> pd.Series:
    """
    TRIX oscillator: 1-period ROC of triple-smoothed EMA.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if drift <= 0:
        raise ValueError("drift must be > 0")

    close_s = check_series(close, "close")
    ema1 = close_s.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()

    result = 100.0 * ema3.pct_change(drift)
    result.name = f"TRIX_{length}_{drift}"
    return result
