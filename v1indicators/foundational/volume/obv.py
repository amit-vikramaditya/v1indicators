import pandas as pd
import numpy as np
from .._utils import check_series

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV)."""
    close = check_series(close, "close")
    volume = check_series(volume, "volume")

    direction = pd.Series(np.sign(close.diff()), index=close.index)
    # If direction is NaN (first), fill 0
    # If direction is 0, volume is ignored (0 * vol = 0)
    
    obv_val = (volume * direction.fillna(0)).cumsum()
    obv_val.name = "OBV"
    return obv_val

