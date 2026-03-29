import pandas as pd

from .._utils import check_series


def cdl_inside(high: pd.Series, low: pd.Series, asbool: bool = False) -> pd.Series:
    """Inside-bar pattern signal."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")

    inside = (high_s < high_s.shift(1)) & (low_s > low_s.shift(1))
    out = inside if asbool else inside.astype(int)
    out.name = "CDL_INSIDE"
    return out
