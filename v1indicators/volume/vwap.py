import pandas as pd


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price (cumulative VWAP)."""

    if not all(isinstance(x, pd.Series) for x in (high, low, close, volume)):
        raise TypeError("high, low, close, volume must be pandas Series")

    typical = (high + low + close) / 3.0

    cum_vol = volume.cumsum()
    cum_pv = (typical * volume).cumsum()

    return cum_pv / cum_vol.replace(0, pd.NA)

