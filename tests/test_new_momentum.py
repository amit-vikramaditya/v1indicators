import pandas as pd
import numpy as np
import pytest
from v1indicators.momentum.mfi import mfi
from v1indicators.momentum.cci import cci

def test_mfi_basic():
    high = pd.Series([10, 11, 12, 13, 14, 15])
    low = pd.Series([8, 9, 10, 11, 12, 13])
    close = pd.Series([9, 10, 11, 12, 13, 14])
    volume = pd.Series([100, 200, 150, 300, 250, 400])
    
    result = mfi(high, low, close, volume, length=2)
    assert isinstance(result, pd.Series)
    assert len(result) == 6
    # In this uptrend, MFI should be high
    assert result.iloc[-1] > 50

def test_cci_basic():
    high = pd.Series([10, 11, 12, 13, 14, 15])
    low = pd.Series([8, 9, 10, 11, 12, 13])
    close = pd.Series([9, 10, 11, 12, 13, 14])
    
    result = cci(high, low, close, length=2)
    assert isinstance(result, pd.Series)
    assert len(result) == 6
    # CCI should be positive in uptrend
    assert result.iloc[-1] > 0
