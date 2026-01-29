import pandas as pd


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    smooth: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D)."""

    if not all(isinstance(x, pd.Series) for x in (high, low, close)):
        raise TypeError("high, low, close must be pandas Series")

    if min(length, smooth) <= 0:
        raise ValueError("length and smooth must be > 0")

    lowest = low.rolling(length).min()
    highest = high.rolling(length).max()

    range_ = highest - lowest
    k = 100 * (close - lowest) / range_.replace(0, pd.NA)
    d = k.rolling(smooth).mean()

    return pd.DataFrame({
        "stoch_k": k,
        "stoch_d": d,
    })

