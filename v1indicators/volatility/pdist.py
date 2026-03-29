import pandas as pd

from .._utils import check_series


def pdist(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, drift: int = 1) -> pd.Series:
    """Price Distance metric."""
    if drift <= 0:
        raise ValueError("drift must be > 0")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    out = 2.0 * (high_s - low_s).abs()
    out = out + (open_s - close_s.shift(drift)).abs() - (close_s - open_s).abs()
    out.name = "PDIST"
    return out
