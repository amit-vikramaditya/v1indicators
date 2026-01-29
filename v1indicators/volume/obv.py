import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV)."""

    if not all(isinstance(x, pd.Series) for x in (close, volume)):
        raise TypeError("close and volume must be pandas Series")

    direction = close.diff().sign()

    return (volume * direction.fillna(0)).cumsum()

