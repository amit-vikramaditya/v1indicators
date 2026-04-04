import numpy as np
import pandas as pd

from .._utils import check_series


def cmo(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Chande Momentum Oscillator.

    CMO = 100 * (sum(gains) - sum(losses)) / (sum(gains) + sum(losses))
    over rolling `length` window.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    delta = close_s.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    sum_g = gains.rolling(length).sum()
    sum_l = losses.rolling(length).sum()
    denom = (sum_g + sum_l).replace(0.0, np.nan)

    out = 100.0 * (sum_g - sum_l) / denom
    out.name = f"CMO_{length}"
    return out
