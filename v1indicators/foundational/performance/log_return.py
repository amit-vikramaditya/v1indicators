import numpy as np
import pandas as pd

from ..._utils import check_series


def log_return(close: pd.Series, cumulative: bool = False, length: int = 1) -> pd.Series:
    """Log return over `length` bars."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close").replace(0.0, np.nan)
    out = np.log(close_s / close_s.shift(length))
    if cumulative:
        out = out.cumsum()
    out.name = f"LOGRET_{length}"
    return out
