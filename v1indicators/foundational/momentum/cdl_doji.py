import pandas as pd

from .._utils import check_series


def cdl_doji(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, factor: float = 0.1) -> pd.Series:
    """Doji candlestick pattern signal."""
    if factor <= 0:
        raise ValueError("factor must be > 0")

    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    body = (close_s - open_s).abs()
    rng = (high_s - low_s).replace(0.0, pd.NA)
    out = (body <= factor * rng).astype(int)
    out.name = "CDL_DOJI"
    return out
