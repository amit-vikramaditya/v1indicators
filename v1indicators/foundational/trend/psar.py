import pandas as pd
import numpy as np
from numba import njit
from .._utils import check_series

@njit
def _psar_kernel(high_v, low_v, acceleration, maximum):
    length = len(high_v)
    psar_v = np.full(length, np.nan)
    psar_dir = np.zeros(length, dtype=np.int8)  # 1 for bull, -1 for bear
    
    if length < 2:
        return psar_v, psar_dir

    # Initial values
    bull = True
    
    # Starting point
    if high_v[1] > high_v[0] or low_v[1] > low_v[0]:
        bull = True
        psar_v[1] = low_v[0]
        ep = high_v[1]
    else:
        bull = False
        psar_v[1] = high_v[0]
        ep = low_v[1]
        
    af = acceleration
    psar_dir[1] = 1 if bull else -1
    
    for i in range(2, length):
        # Calculate PSAR for current bar
        prev_psar = psar_v[i-1]
        psar_v[i] = prev_psar + af * (ep - prev_psar)
        
        # Ensure PSAR doesn't enter the previous two bars' range
        if bull:
            if psar_v[i] > low_v[i] or psar_v[i] > low_v[i-1]:
                # Switch to Bear
                bull = False
                psar_v[i] = ep  # New PSAR is the previous EP
                ep = low_v[i]
                af = acceleration
            else:
                # Continue Bull
                if high_v[i] > ep:
                    ep = high_v[i]
                    af = min(af + acceleration, maximum)
                # Cap PSAR so it's not higher than previous lows
                psar_v[i] = min(psar_v[i], low_v[i-1], low_v[i-2])
        else:
            if psar_v[i] < high_v[i] or psar_v[i] < high_v[i-1]:
                # Switch to Bull
                bull = True
                psar_v[i] = ep
                ep = high_v[i]
                af = acceleration
            else:
                # Continue Bear
                if low_v[i] < ep:
                    ep = low_v[i]
                    af = min(af + acceleration, maximum)
                # Cap PSAR so it's not lower than previous highs
                psar_v[i] = max(psar_v[i], high_v[i-1], high_v[i-2])
                
        psar_dir[i] = 1 if bull else -1
        
    return psar_v, psar_dir

def psar(
    high: pd.Series,
    low: pd.Series,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> pd.DataFrame:
    """
    Parabolic Stop and Reverse (PSAR). (Numba Optimized)
    
    A trend-following indicator that uses an accelerating factor to trail stops.
    """
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    
    high_v = high_s.to_numpy()
    low_v = low_s.to_numpy()
    
    psar_v, psar_dir = _psar_kernel(high_v, low_v, acceleration, maximum)

    return pd.DataFrame({
        "PSAR": psar_v,
        "PSAR_DIR": psar_dir
    }, index=high_s.index)
