import pandas as pd
import numpy as np
import pytest
from v1indicators.overlap.ema import ema

def test_ema_basic():
    """Test EMA with simple data."""
    data = pd.Series([10.0, 11.0, 12.0, 13.0])
    # length=2. alpha = 2/(2+1) = 2/3 = 0.666...
    # But wait, standard EMA formula usually uses alpha = 2/(N+1).
    # Pandas span=N uses alpha = 2/(N+1).
    
    # sma[0] (or seed) = 10 (if adjust=False, usually first value is seed)
    # ema[1] = alpha * 11 + (1-alpha) * 10
    
    result = ema(data, length=2)
    assert isinstance(result, pd.Series)
    assert len(result) == 4
    
    # Check values roughly
    # 1st value: 10.0
    # 2nd value: (2/3)*11 + (1/3)*10 = 7.33 + 3.33 = 10.66...
    
    # Let's verify against pandas directly
    expected = data.ewm(span=2, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_ema_length_1():
    """EMA(1) should be identity."""
    data = pd.Series([1.0, 2.0, 3.0])
    result = ema(data, length=1)
    pd.testing.assert_series_equal(result, data, check_names=False)
