import numpy as np
import pandas as pd

from .._utils import check_series


def ui(close: pd.Series, length: int = 14, scalar: float = 100.0) -> pd.Series:
    """Ulcer Index."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    highest = close_s.rolling(length).max().replace(0.0, np.nan)
    downside = scalar * (close_s - highest) / highest
    out = np.sqrt((downside.pow(2)).rolling(length).sum() / float(length))
    out.name = f"UI_{length}"
    return out
