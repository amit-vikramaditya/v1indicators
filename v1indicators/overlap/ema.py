# v1indicators/overlap/ema.py

import pandas as pd

def ema(close: pd.Series, length: int, adjust: bool = False) -> pd.Series:
    """
    Exponential Moving Average (EMA)

    Parameters
    ----------
    close : pd.Series
        Series of closing prices
    length : int
        EMA period
    adjust : bool, default False
        Same behavior as pandas ewm(adjust=...)

    Returns
    -------
    pd.Series
        EMA values
    """
    if not isinstance(close, pd.Series):
        raise TypeError("ema() expects a pandas Series")

    if len(close) < length:
        return pd.Series(index=close.index, dtype="float64")

    return close.ewm(span=length, adjust=adjust).mean()

