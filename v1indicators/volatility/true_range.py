import pandas as pd

from .._utils import check_series


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range using high, low, and previous close."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    prev_close = close_s.shift(1)
    tr = pd.concat(
        [high_s - low_s, (high_s - prev_close).abs(), (low_s - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.name = "TRUERANGE_1"
    return tr
