import pandas as pd

from .._utils import check_series


def trend_return(close: pd.Series, length: int = 20, cumulative: bool = True) -> pd.Series:
    """Trend return over lookback windows."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    tr = close_s.pct_change(length)
    out = (1.0 + tr).cumprod() - 1.0 if cumulative else tr
    out.name = f"TREND_RETURN_{length}"
    return out
