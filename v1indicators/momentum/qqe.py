import pandas as pd

from .._utils import check_series
from .rsi import rsi


def qqe(close: pd.Series, length: int = 14, smooth: int = 5, factor: float = 4.236) -> pd.DataFrame:
    """QQE approximation with dynamic bands around smoothed RSI."""
    if min(length, smooth) <= 0:
        raise ValueError("length and smooth must be > 0")
    if factor <= 0:
        raise ValueError("factor must be > 0")

    close_s = check_series(close, "close")
    r = rsi(close_s, length=length)
    rs = r.ewm(span=smooth, adjust=False).mean()
    dr = rs.diff().abs().ewm(span=length, adjust=False).mean()
    dar = factor * dr

    upper = rs + dar
    lower = rs - dar
    return pd.DataFrame({"QQE": rs, "QQE_UPPER": upper, "QQE_LOWER": lower})
