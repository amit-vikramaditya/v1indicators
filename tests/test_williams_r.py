import pandas as pd
import pytest

from v1indicators.momentum.williams_r import williams_r


def test_williams_r_basic():
    high = pd.Series([12.0, 13.0, 14.0, 15.0, 16.0, 15.0])
    low = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0, 11.0])
    close = pd.Series([10.0, 12.0, 13.0, 14.0, 15.0, 13.0])

    result = williams_r(high, low, close, length=3)

    highest = high.rolling(3).max()
    lowest = low.rolling(3).min()
    expected = -100.0 * (highest - close) / (highest - lowest)
    expected.name = "WILLR_3"

    pd.testing.assert_series_equal(result, expected)


def test_williams_r_flat_range_nan():
    high = pd.Series([10.0, 10.0, 10.0, 10.0])
    low = pd.Series([10.0, 10.0, 10.0, 10.0])
    close = pd.Series([10.0, 10.0, 10.0, 10.0])

    result = williams_r(high, low, close, length=2)
    assert result.isna().all()


def test_williams_r_input_validation():
    with pytest.raises(ValueError):
        williams_r(pd.Series([1.0]), pd.Series([1.0]), pd.Series([1.0]), length=0)

    with pytest.raises(TypeError):
        williams_r([1.0], pd.Series([1.0]), pd.Series([1.0]), length=2)
