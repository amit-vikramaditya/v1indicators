import pandas as pd


def donchian(
    high: pd.Series,
    low: pd.Series,
    length: int = 20,
) -> pd.DataFrame:
    """Donchian Channels."""

    if not all(isinstance(x, pd.Series) for x in (high, low)):
        raise TypeError("high and low must be pandas Series")

    if length <= 0:
        raise ValueError("length must be > 0")

    upper = high.rolling(length).max()
    lower = low.rolling(length).min()
    mid = (upper + lower) / 2

    return pd.DataFrame({
        "donchian_upper": upper,
        "donchian_mid": mid,
        "donchian_lower": lower,
    })

