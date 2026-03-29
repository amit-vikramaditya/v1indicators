import pandas as pd

from .._utils import check_series


def percent_return(close: pd.Series, cumulative: bool = False, length: int = 1) -> pd.Series:
    """Percent return over `length` bars."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    out = close_s.pct_change(length)
    if cumulative:
        out = (1.0 + out).cumprod() - 1.0
    out.name = f"PCTRET_{length}"
    return out
