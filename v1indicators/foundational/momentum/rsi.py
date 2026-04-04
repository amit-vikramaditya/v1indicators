import pandas as pd
import numpy as np
from .._utils import check_series

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    A momentum oscillator that measures the speed and change of price movements
    using Wilder's smoothing (RMA).

    Formula:
        RSI = 100 - [100 / (1 + RS)]
        RS = AvgGain / AvgLoss

    Args:
        close: Pandas Series of prices.
        length: Period length (default 14).

    Returns:
        Pandas Series named 'RSI_{length}'.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close = check_series(close, "close")
    delta = close.diff()

    positive = delta.copy()
    negative = delta.copy()
    positive[positive < 0] = 0.0
    negative[negative > 0] = 0.0

    # Wilder alpha with explicit warmup period.
    alpha = 1.0 / float(length)
    avg_gain = positive.ewm(alpha=alpha, min_periods=length, adjust=True).mean()
    avg_loss = negative.ewm(alpha=alpha, min_periods=length, adjust=True).mean().abs()

    with np.errstate(divide="ignore", invalid="ignore"):
        rsi_series = 100.0 * avg_gain / (avg_gain + avg_loss)

    rsi_series.name = f"RSI_{length}"
    return rsi_series

