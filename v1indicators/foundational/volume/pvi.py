import pandas as pd

from .._utils import check_series


def pvi(close: pd.Series, volume: pd.Series, length: int = 1, initial: float = 100.0) -> pd.Series:
    """Positive Volume Index."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if initial <= 0:
        raise ValueError("initial must be > 0")

    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    roc = close_s.pct_change(length).fillna(0.0)
    trigger = (volume_s.diff() > 0.0).astype(float)
    contrib = trigger * roc
    out = (1.0 + contrib).cumprod() * initial
    out.name = f"PVI_{length}"
    return out
