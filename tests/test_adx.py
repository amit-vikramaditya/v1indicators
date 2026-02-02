import pandas as pd
from v1indicators.trend.adx import adx

def test_adx_basic():
    """Test ADX with synthetic trend data."""
    # Strong uptrend
    high = pd.Series([10, 12, 14, 16, 18, 20, 22, 24])
    low = pd.Series([ 8, 10, 12, 14, 16, 18, 20, 22])
    close = pd.Series([9, 11, 13, 15, 17, 19, 21, 23])
    
    # Length 2 for short test
    result = adx(high, low, close, length=2)
    
    assert isinstance(result, pd.DataFrame)
    assert 'ADX_2' in result.columns
    assert 'DMP_2' in result.columns
    assert 'DMN_2' in result.columns
    
    # In a perfect uptrend:
    # UpMove = 2, DownMove = -(-2) = 2?? No.
    # High diff = 2. Low diff = 2.
    # UpMove = 2. DownMove = 0 (since Low went up, diff is positive, -diff is negative. DownMove is max(-diff, 0) logic usually).
    # Wait, DownMove logic:
    # down_move = -low.diff()
    # If low increases (8->10), diff is 2. down_move is -2.
    # DM- is keep if > DM+ and > 0.
    # So in strict uptrend, DM- should be 0.
    # So Plus_DI should be high, Minus_DI should be 0.
    
    # Skip first few NaN
    last_row = result.iloc[-1]
    assert last_row['DMP_2'] > last_row['DMN_2']
    assert last_row['ADX_2'] > 50  # Strong trend
