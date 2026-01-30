import pandas as pd
import numpy as np

def wma(close: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be pandas Series")

    if length <= 0:
        raise ValueError("length must be > 0")

    weights = np.arange(1, length + 1)
    weights_sum = weights.sum()

    def linear_wma(x):
        return np.dot(x, weights) / weights_sum

    return close.rolling(length).apply(linear_wma, raw=True)

