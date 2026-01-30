import pandas as pd

def roc(close: pd.Series, length: int = 12) -> pd.Series:
    """Rate of Change ((Price / Price[n]) - 1) * 100."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be pandas Series")

    if length <= 0:
        raise ValueError("length must be > 0")

    prev_close = close.shift(length)
    return ((close / prev_close) - 1) * 100

