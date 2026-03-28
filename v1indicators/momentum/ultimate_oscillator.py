import numpy as np
import pandas as pd

from .._utils import check_series


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short: int = 7,
    medium: int = 14,
    long: int = 28,
    w_short: float = 4.0,
    w_medium: float = 2.0,
    w_long: float = 1.0,
) -> pd.Series:
    """
    Ultimate Oscillator (UO).
    """
    if short <= 0 or medium <= 0 or long <= 0:
        raise ValueError("periods must be > 0")
    if not (short < medium < long):
        raise ValueError("periods must satisfy short < medium < long")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    prev_close = close_s.shift(1)
    min_low_or_prev = pd.concat([low_s, prev_close], axis=1).min(axis=1)
    max_high_or_prev = pd.concat([high_s, prev_close], axis=1).max(axis=1)

    bp = close_s - min_low_or_prev
    tr = (max_high_or_prev - min_low_or_prev).replace(0.0, np.nan)

    avg_short = bp.rolling(short).sum() / tr.rolling(short).sum()
    avg_medium = bp.rolling(medium).sum() / tr.rolling(medium).sum()
    avg_long = bp.rolling(long).sum() / tr.rolling(long).sum()

    weight_sum = w_short + w_medium + w_long
    result = 100.0 * (
        w_short * avg_short + w_medium * avg_medium + w_long * avg_long
    ) / weight_sum
    result.name = f"UO_{short}_{medium}_{long}"
    return result
