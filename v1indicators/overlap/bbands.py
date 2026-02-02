import pandas as pd
from .._utils import check_series

def bbands(
    close: pd.Series,
    length: int = 20,
    mult: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands."""
    
    if length <= 0:
        raise ValueError("length must be > 0")

    close = check_series(close, "close")

    # Use rolling window for calculation
    # min_periods=length to ensure we don't return partial data if not desired?
    # Standard usually requires full window.
    roller = close.rolling(length)
    
    mid = roller.mean()
    std = roller.std(ddof=1) # Sample standard deviation (n-1)

    upper = mid + std * mult
    lower = mid - std * mult

    return pd.DataFrame({
        "BB_LOWER": lower,
        "BB_MID": mid,
        "BB_UPPER": upper,
    })