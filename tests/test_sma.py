import pandas as pd
import numpy as np
import pytest
from v1indicators.overlap.sma import sma

def test_sma_basic():
    """Test SMA with a simple linear series."""
    data = pd.Series([1, 2, 3, 4, 5])
    result = sma(data, length=3)
    
    # Expected: [NaN, NaN, 2.0, 3.0, 4.0]
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_sma_input_validation():
    """Test SMA input validation."""
    with pytest.raises(ValueError):
        sma(pd.Series([1, 2, 3]), length=0)
    
    with pytest.raises(TypeError):
        sma([1, 2, 3], length=3)
