import pandas as pd

from .._utils import check_series


def qstick(open_: pd.Series, close: pd.Series, length: int = 10) -> pd.Series:
    """Qstick = SMA(close - open, length)."""
    if length <= 0:
        raise ValueError("length must be > 0")

    open_s = check_series(open_, "open")
    close_s = check_series(close, "close")
    out = (close_s - open_s).rolling(length).mean()
    out.name = f"QSTICK_{length}"
    return out
