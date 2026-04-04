import pandas as pd
from .._utils import check_series

def donchian(
    high: pd.Series,
    low: pd.Series,
    length: int = 20,
) -> pd.DataFrame:
    """Donchian Channels."""
    
    if length <= 0:
        raise ValueError("length must be > 0")

    high = check_series(high, "high")
    low = check_series(low, "low")

    upper = high.rolling(length).max()
    lower = low.rolling(length).min()
    mid = (upper + lower) / 2

    return pd.DataFrame({
        "DONCHIAN_UPPER": upper,
        "DONCHIAN_MID": mid,
        "DONCHIAN_LOWER": lower,
    })

