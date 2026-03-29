import numpy as np
import pandas as pd

from .._utils import check_series


def wavetrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    channel_length: int = 10,
    average_length: int = 21,
    signal_length: int = 4,
) -> pd.DataFrame:
    """
    WaveTrend oscillator.

    Returns the main line (WT1), signal line (WT2), and histogram.
    """
    if channel_length <= 0 or average_length <= 0 or signal_length <= 0:
        raise ValueError("channel_length, average_length, and signal_length must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    ap = (high_s + low_s + close_s) / 3.0
    esa = ap.ewm(span=channel_length, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=channel_length, adjust=False).mean().replace(0.0, np.nan)
    ci = (ap - esa) / (0.015 * d)

    wt1 = ci.ewm(span=average_length, adjust=False).mean()
    wt2 = wt1.rolling(signal_length).mean()
    hist = wt1 - wt2

    return pd.DataFrame(
        {
            "WT1": wt1,
            "WT2": wt2,
            "WT_HIST": hist,
        }
    )
