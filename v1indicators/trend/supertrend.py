import pandas as pd
import numpy as np
from ..volatility.atr import atr
from .._utils import check_series

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    mult: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend Indicator.

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

    high = check_series(high, "high")
    low = check_series(low, "low")
    close = check_series(close, "close")

    # Calculate ATR (returns Series)
    atr_v = atr(high, low, close, length)
    
    # Pre-calculate bands (vectorized)
    hl2 = (high + low) / 2
    upper_band_basic = hl2 + mult * atr_v
    lower_band_basic = hl2 - mult * atr_v
    
    # Convert to NumPy for the loop (Critical for performance)
    close_val = close.values
    upper_basic = upper_band_basic.values
    lower_basic = lower_band_basic.values
    
    length_data = len(close)
    st = np.full(length_data, np.nan)
    direction = np.full(length_data, 0, dtype=np.int8)
    
    # Initialize arrays for the recursive calculation
    final_upper = np.copy(upper_basic)
    final_lower = np.copy(lower_basic)
    
    # Initialize first valid index (skip NaNs from ATR)
    # ATR has first (length-1) NaNs? No, Wilder's RMA has valid values after start if we treat them right, 
    # but my rma implementation uses ewm(adjust=False), so it starts immediately.
    # However, conventionally we might start logic later. 
    # Let's iterate from index 1.
    
    # Logic:
    # If Close > Previous Final Upper -> Uptrend (Dir=1)
    # If Close < Previous Final Lower -> Downtrend (Dir=-1)
    # Else -> Same Dir
    
    # Refined loop logic to match standard Supertrend
    # We need to maintain the "Final" bands state.
    
    # Assumption: Start with direction=1 or based on first comparison?
    # Standard usually defaults to 1 or -1 based on Close vs Band.
    
    # Let's perform the loop.
    # Warning: upper_basic/lower_basic might contain NaNs at start.
    
    current_dir = 1
    
    for i in range(1, length_data):
        if np.isnan(upper_basic[i]):
            continue
            
        # Recursive bands logic
        # If trend is up (1), Lower Band = max(Basic Lower, Prev Final Lower)
        # If trend is down (-1), Upper Band = min(Basic Upper, Prev Final Upper)
        
        prev_upper = final_upper[i-1]
        prev_lower = final_lower[i-1]
        
        # 1. Update Final Bands based on previous close
        # (Wait, standard logic uses previous Close to determine if we can tighten?)
        
        # Standard Supertrend Logic:
        # Final Lower = Basic Lower
        # If (Basic Lower < Prev Final Lower) and (Prev Close > Prev Final Lower):
        #    Final Lower = Prev Final Lower
        
        if (lower_basic[i] < prev_lower) and (close_val[i-1] > prev_lower):
            final_lower[i] = prev_lower
            
        if (upper_basic[i] > prev_upper) and (close_val[i-1] < prev_upper):
            final_upper[i] = prev_upper
            
        # 2. Determine Direction
        # If Close > Prev Final Upper -> switch to 1 (Uptrend)
        # If Close < Prev Final Lower -> switch to -1 (Downtrend)
        
        if close_val[i] > prev_upper:
            current_dir = 1
        elif close_val[i] < prev_lower:
            current_dir = -1
        
        direction[i] = current_dir
        
        # 3. Set Supertrend Value
        if current_dir == 1:
            st[i] = final_lower[i]
        else:
            st[i] = final_upper[i]

    return pd.DataFrame({
        "SUPERTREND": st,
        "SUPERTREND_DIR": direction,
    }, index=close.index)

