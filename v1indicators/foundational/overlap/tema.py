import pandas as pd

from .._utils import check_series


def tema(close: pd.Series, length: int = 20) -> pd.Series:
    """
    Triple Exponential Moving Average (TEMA).

    TEMA = 3*EMA1 - 3*EMA2 + EMA3
    where EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    ema1 = close_s.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()

    result = 3.0 * ema1 - 3.0 * ema2 + ema3
    result.name = f"TEMA_{length}"
    return result
