import pandas as pd
import pytest
from v1indicators.trend.psar import psar

def test_psar_basic():
    """Test PSAR with a clear trend switch."""
    # Price starts at 10 and goes up, then down
    high = pd.Series([10, 11, 12, 13, 14, 13, 12, 11, 10])
    low = pd.Series([ 8,  9, 10, 11, 12, 11, 10,  9,  8])
    
    result = psar(high, low)
    
    assert isinstance(result, pd.DataFrame)
    assert "PSAR" in result.columns
    assert "PSAR_DIR" in result.columns
    
    # Check that direction actually changes
    unique_dirs = result["PSAR_DIR"].unique()
    assert 1 in unique_dirs
    assert -1 in unique_dirs
    
    # Check lengths match
    assert len(result) == len(high)

def test_psar_no_data():
    high = pd.Series([10])
    low = pd.Series([8])
    result = psar(high, low)
    assert result["PSAR"].isna().all()
