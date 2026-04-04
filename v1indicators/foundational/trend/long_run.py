import pandas as pd

from .._utils import check_series


def long_run(fast: pd.Series, slow: pd.Series, length: int = 2) -> pd.Series:
    """Long-run state: fast > slow for rolling length."""
    if length <= 0:
        raise ValueError("length must be > 0")

    f = check_series(fast, "fast")
    s = check_series(slow, "slow")
    cond = (f > s).astype(float).rolling(length).sum() == float(length)
    out = cond.astype(int)
    out.name = f"LONG_RUN_{length}"
    return out
