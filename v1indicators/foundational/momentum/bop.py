import numpy as np
import pandas as pd

from .._utils import check_series


def bop(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Balance Of Power = (close - open) / (high - low)."""
    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    denom = (high_s - low_s).replace(0.0, np.nan)
    out = (close_s - open_s) / denom
    out.name = "BOP"
    return out
