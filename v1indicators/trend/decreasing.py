import pandas as pd

from .._utils import check_series


def decreasing(close: pd.Series, length: int = 1, strict: bool = False, percent: float = 0.0) -> pd.Series:
    """Boolean series for decreasing behavior over lookback length."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    diff = close_s.diff(length)

    if percent > 0.0:
        thresh = close_s.shift(length).abs() * (percent / 100.0)
        base = diff < -thresh
    else:
        base = diff < 0.0

    if strict:
        step = close_s.diff() < 0.0
        out = step.rolling(length).sum() == float(length)
    else:
        out = base

    out = out.astype(int)
    out.name = f"DECREASING_{length}"
    return out
