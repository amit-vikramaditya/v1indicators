import numpy as np
import pandas as pd

from .._utils import check_series


def coppock(close: pd.Series, long: int = 14, short: int = 11, smooth: int = 10) -> pd.Series:
    """Coppock Curve = WMA(ROC(long) + ROC(short), smooth)."""
    if min(long, short, smooth) <= 0:
        raise ValueError("long, short, and smooth must be > 0")

    close_s = check_series(close, "close")

    roc_long = 100.0 * (close_s / close_s.shift(long) - 1.0)
    roc_short = 100.0 * (close_s / close_s.shift(short) - 1.0)
    raw = roc_long + roc_short

    w = np.arange(1, smooth + 1, dtype=np.float64)
    out = raw.rolling(smooth).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    out.name = f"COPPOCK_{long}_{short}_{smooth}"
    return out
