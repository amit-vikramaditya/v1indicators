import pandas as pd

from .._utils import check_series


def psl(close: pd.Series, length: int = 12) -> pd.Series:
    """Psychological Line: percent of up closes in rolling window."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    up = (close_s.diff() > 0.0).astype(float)

    out = 100.0 * up.rolling(length).mean()
    out.name = f"PSL_{length}"
    return out
