import pandas as pd
import pytest

from v1indicators.trend.zigzag_swings import zigzag_swings


def test_zigzag_swings_basic_shape():
    high = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0])
    low = pd.Series([8.0, 9.0, 8.5, 9.5, 9.0, 10.0, 9.4])

    result = zigzag_swings(high, low, length=1)
    assert list(result.columns) == ["SWING_HIGH", "SWING_LOW", "ZZ_TREND"]
    assert len(result) == len(high)


def test_zigzag_swings_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        zigzag_swings(s, s, length=0)
    with pytest.raises(TypeError):
        zigzag_swings([1.0, 2.0], s)
