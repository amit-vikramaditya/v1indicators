import pandas as pd
import numpy as np
import pytest
from v1indicators.momentum.macd import macd

def test_macd_basic():
    """Test MACD with simple data."""
    data = pd.Series(np.random.randn(100) + 100)
    
    result = macd(data, fast=12, slow=26, signal=9)
    
    assert isinstance(result, pd.DataFrame)
    assert 'MACD' in result.columns
    assert 'MACD_SIGNAL' in result.columns
    assert 'MACD_HIST' in result.columns
    
    # Check lengths
    assert len(result) == 100
    
    # First few should be NaN (at least until slow=26)
    # Actually ema(26) is valid from index 0 (with adjust=False/True depending, but we use adjust=False in wrapper)
    # wait, my ema wrapper uses adjust=False.
    # Pandas ewm(adjust=False) starts immediately with the first value as seed.
    # So MACD should produce values immediately, though they are not "converged".
    # Standard TA-Lib might behave differently.
    # But mathematically, it returns values.
    
    assert not result['MACD'].isna().all()

def test_macd_input_validation():
    with pytest.raises(ValueError):
        macd(pd.Series([1,2]), fast=0)
