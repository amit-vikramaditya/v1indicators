import pandas as pd

from .._utils import check_series


def eri(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 13) -> pd.DataFrame:
    """Elder Ray Index (bull and bear power)."""
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    ema = close_s.ewm(span=length, adjust=False).mean()
    bull = high_s - ema
    bear = low_s - ema

    return pd.DataFrame({f"BULLP_{length}": bull, f"BEARP_{length}": bear})
