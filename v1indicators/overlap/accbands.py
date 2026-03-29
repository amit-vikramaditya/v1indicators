import pandas as pd

from .._utils import check_series


def accbands(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, c: float = 4.0) -> pd.DataFrame:
    """Acceleration Bands."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if c <= 0:
        raise ValueError("c must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    hl_sum = (high_s + low_s).replace(0.0, pd.NA)
    adj = c * (high_s - low_s) / hl_sum
    upper = (high_s * (1.0 + adj)).rolling(length).mean()
    lower = (low_s * (1.0 - adj)).rolling(length).mean()
    mid = close_s.rolling(length).mean()

    return pd.DataFrame({f"ACCBL_{length}": lower, f"ACCBM_{length}": mid, f"ACCBU_{length}": upper})
