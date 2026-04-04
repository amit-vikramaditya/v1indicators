import pandas as pd
import pytest

from v1indicators.levels import equal_highs_lows


def test_equal_highs_lows_basic():
    high = pd.Series([10.0, 12.0, 11.0, 12.01, 11.5, 10.8, 11.0])
    low = pd.Series([9.0, 9.2, 8.8, 9.1, 8.79, 8.7, 8.8])

    result = equal_highs_lows(high, low, length=1, threshold=0.02)
    assert "EQUAL_HIGH" in result.columns
    assert "EQUAL_LOW" in result.columns
    assert result["EQUAL_HIGH"].dtype == bool
    assert result["EQUAL_LOW"].dtype == bool


def test_equal_highs_lows_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        equal_highs_lows(s, s, length=0)
    with pytest.raises(ValueError):
        equal_highs_lows(s, s, threshold=-0.1)
    with pytest.raises(TypeError):
        equal_highs_lows([1.0, 2.0], s)
