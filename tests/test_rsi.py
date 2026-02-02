import pandas as pd
import numpy as np
import pytest
from v1indicators.momentum.rsi import rsi

def test_rsi_basic():
    """Test RSI with a known sequence."""
    # A sequence with some up and down movement
    data = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0])
    # length=2 for short test
    result = rsi(data, length=2)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(data)
    # First value is always NaN because of diff()
    assert np.isnan(result[0])

def test_rsi_constant():
    """Test RSI with constant prices (should be 0 change -> NaN or handled)."""
    data = pd.Series([10.0] * 10)
    result = rsi(data, length=2)
    
    # If prices are constant, diff is 0.
    # gain=0, loss=0. avg_gain=0, avg_loss=0.
    # rs = 0/0 = NaN.
    # RSI = NaN.
    # This is mathematically correct for "No Momentum".
    assert result.isna().all()

def test_rsi_zero_loss():
    """Test RSI when price only goes up (Loss = 0) -> RSI should be 100."""
    data = pd.Series([10, 11, 12, 13, 14, 15])
    result = rsi(data, length=2)
    # First index is NaN (diff)
    # Subsequent should approach 100
    assert result.iloc[-1] == 100.0
