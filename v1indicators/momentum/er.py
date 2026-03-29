import numpy as np
import pandas as pd

from .._utils import check_series


def er(close: pd.Series, length: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    change = (close_s - close_s.shift(length)).abs()
    volatility = close_s.diff().abs().rolling(length).sum().replace(0.0, np.nan)

    out = change / volatility
    out.name = f"ER_{length}"
    return out
