import pandas as pd

from .._utils import check_series


def quantile(close: pd.Series, length: int = 14, q: float = 0.5) -> pd.Series:
    """Rolling quantile."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")

    close_s = check_series(close, "close")
    out = close_s.rolling(length).quantile(q)
    out.name = f"QUANTILE_{length}_{q}"
    return out
