import pandas as pd
import numpy as np
from .._utils import check_series

def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    c: float = 0.015,
) -> pd.Series:
    """
    Commodity Channel Index (CCI).
    Measures the deviation of price from its statistical average.
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")

    typical_price = (high + low + close) / 3.0
    
    # SMA of Typical Price
    sma_tp = typical_price.rolling(length).mean()
    
    # Mean Absolute Deviation (MAD)
    # MAD = mean(|TP - SMA(TP)|)
    def mad(x):
        return np.abs(x - x.mean()).mean()
    
    mean_dev = typical_price.rolling(length).apply(mad, raw=True)
    
    # CCI Formula: (TP - SMA_TP) / (0.015 * MAD)
    cci_val = (typical_price - sma_tp) / (c * mean_dev)
    
    cci_val.name = f"CCI_{length}"
    return cci_val
