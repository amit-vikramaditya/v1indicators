import pandas as pd
from .._utils import check_series

def ema(close: pd.Series, length: int, adjust: bool = False) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    A weighted moving average that gives more weight to recent price data.

    Formula:
        EMA_t = (Price_t * alpha) + (EMA_{t-1} * (1 - alpha))
        alpha = 2 / (length + 1)

    Args:
        close: Pandas Series of prices.
        length: Number of periods.
        adjust: If True, uses Pandas' legacy pre-mean adjustment (default False).

    Returns:
        Pandas Series named 'EMA_{length}'.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    series = check_series(close, "close")
    
    result = series.ewm(span=length, adjust=adjust).mean()
    result.name = f"EMA_{length}"
    return result

