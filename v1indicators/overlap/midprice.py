import pandas as pd

from .._utils import check_series


def midprice(high: pd.Series, low: pd.Series, length: int = 14) -> pd.Series:
    """Midprice of rolling highest high and lowest low."""
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    out = 0.5 * (high_s.rolling(length).max() + low_s.rolling(length).min())
    out.name = f"MIDPRICE_{length}"
    return out
