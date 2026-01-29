import pandas as pd

def ema(close: pd.Series, length: int, adjust: bool = False) -> pd.Series:
    """Exponential Moving Average."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be pandas Series")

    if length <= 0:
        raise ValueError("length must be > 0")

    return close.ewm(span=length, adjust=adjust).mean()

