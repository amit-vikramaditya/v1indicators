import pandas as pd
from .._utils import check_series

def roc(close: pd.Series, length: int = 12) -> pd.Series:
    """Rate of Change ((Price / Price[n]) - 1) * 100."""
    if length <= 0:
        raise ValueError("length must be > 0")

    close = check_series(close, "close")
    
    prev_close = close.shift(length)
    roc_val = ((close / prev_close) - 1) * 100
    roc_val.name = f"ROC_{length}"
    return roc_val

