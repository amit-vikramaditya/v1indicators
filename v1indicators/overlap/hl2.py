import pandas as pd

from .._utils import check_series


def hl2(high: pd.Series, low: pd.Series) -> pd.Series:
    """HL2 = (high + low) / 2."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    out = 0.5 * (high_s + low_s)
    out.name = "HL2"
    return out
