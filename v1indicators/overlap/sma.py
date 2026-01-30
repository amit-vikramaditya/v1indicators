import pandas as pd

def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be pandas Series")

    if length <= 0:
        raise ValueError("length must be > 0")

    return close.rolling(length).mean()
