import pandas as pd
from v1indicators.volatility.atr import atr

def test_atr_basic():
    """Test ATR with simple data."""
    high = pd.Series([10, 12, 11, 13])
    low = pd.Series([8, 9, 7, 10])
    close = pd.Series([9, 11, 10, 12])
    
    # TR: [NaN, max(3, 1, 1)=3, max(4, 0, 1)=4, max(3, 3, 0)=3]
    # Actually Wilder TR starts with High-Low for the first bar
    # tr[0] = 10-8 = 2
    # tr[1] = max(12-9, 12-9, 9-9) = 3
    # tr[2] = max(11-7, 11-11, 7-11) = 4
    # tr[3] = max(13-10, 13-10, 10-10) = 3
    
    result = atr(high, low, close, length=2)
    assert isinstance(result, pd.Series)
    assert len(result) == 4
    # Wilder EMA logic:
    # First value is just TR
    # Next is alpha * tr + (1-alpha) * prev_ema
    # For length 2, alpha = 1/2 = 0.5
