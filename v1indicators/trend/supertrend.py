import pandas as pd
import numpy as np
from numba import njit
from ..volatility.atr import atr
from .._utils import check_series

@njit
def _supertrend_kernel(close_val, upper_basic, lower_basic):
    length_data = len(close_val)
    st = np.full(length_data, np.nan)
    direction = np.full(length_data, 1, dtype=np.int8)
    
    final_upper = np.copy(upper_basic)
    final_lower = np.copy(lower_basic)
    
    current_dir = 1
    
    # Find first non-NaN index
    start_idx = 0
    for i in range(length_data):
        if not np.isnan(upper_basic[i]):
            start_idx = i
            break
    
    if start_idx == 0 and np.isnan(upper_basic[0]):
        return st, direction

    # Initialize first valid supertrend value
    st[start_idx] = final_upper[start_idx] if current_dir == -1 else final_lower[start_idx]

    for i in range(start_idx + 1, length_data):
        prev_upper = final_upper[i-1]
        prev_lower = final_lower[i-1]
        
        # 1. Update Final Bands
        if (lower_basic[i] < prev_lower) and (close_val[i-1] > prev_lower):
            final_lower[i] = prev_lower
            
        if (upper_basic[i] > prev_upper) and (close_val[i-1] < prev_upper):
            final_upper[i] = prev_upper
            
        # 2. Determine Direction
        if close_val[i] > final_upper[i-1]:
            current_dir = 1
        elif close_val[i] < final_lower[i-1]:
            current_dir = -1
        
        direction[i] = current_dir
        
        # 3. Set Supertrend Value
        if current_dir == 1:
            st[i] = final_lower[i]
        else:
            st[i] = final_upper[i]
            
    return st, direction

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    mult: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend Indicator. (Numba Optimized)

    A trend-following indicator that plots a line above or below the price
    based on the Average True Range (ATR). It switches direction when price
    closes on the other side of the line.

    Formula:
        Basic Upper = (High + Low) / 2 + (Multiplier * ATR)
        Basic Lower = (High + Low) / 2 - (Multiplier * ATR)
        (Recursive logic applied to clamp bands based on trend direction)

    Args:
        high: Pandas Series of high prices.
        low: Pandas Series of low prices.
        close: Pandas Series of close prices.
        length: ATR period (default 10).
        mult: ATR multiplier (default 3.0).

    Returns:
        Pandas DataFrame with columns: ['SUPERTREND', 'SUPERTREND_DIR'].
        SUPERTREND_DIR is 1 (Uptrend) or -1 (Downtrend).
    """
    
    if length <= 0 or mult <= 0:
        raise ValueError("length and mult must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    # Calculate ATR (returns Series)
    atr_v = atr(high_s, low_s, close_s, length)
    
    # Pre-calculate bands (vectorized)
    hl2 = (high_s + low_s) / 2
    upper_band_basic = hl2 + mult * atr_v
    lower_band_basic = hl2 - mult * atr_v
    
    # Convert to NumPy for the kernel
    close_val = close_s.to_numpy()
    upper_basic = upper_band_basic.to_numpy()
    lower_basic = lower_band_basic.to_numpy()
    
    st, direction = _supertrend_kernel(close_val, upper_basic, lower_basic)

    return pd.DataFrame({
        "SUPERTREND": st,
        "SUPERTREND_DIR": direction,
    }, index=close.index)

