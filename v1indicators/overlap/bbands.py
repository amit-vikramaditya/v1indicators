import pandas as pd


def bbands(
    close: pd.Series,
    length: int = 20,
    mult: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands."""

    if not isinstance(close, pd.Series):
        raise TypeError("close must be pandas Series")

    if length <= 0:
        raise ValueError("length must be > 0")

    mid = close.rolling(length).mean()
    std = close.rolling(length).std()

    upper = mid + std * mult
    lower = mid - std * mult

    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
    })

