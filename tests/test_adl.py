import pandas as pd
import pytest

from v1indicators.volume.adl import adl


def test_adl_basic():
    high = pd.Series([12.0, 13.0, 14.0, 15.0, 16.0])
    low = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0])
    close = pd.Series([10.0, 12.0, 13.0, 14.0, 15.0])
    volume = pd.Series([100.0, 120.0, 110.0, 130.0, 140.0])

    result = adl(high, low, close, volume)

    mfm = ((close - low) - (high - close)) / (high - low)
    expected = (mfm.fillna(0.0) * volume).cumsum()
    expected.name = "ADL"

    pd.testing.assert_series_equal(result, expected)


def test_adl_flat_range_zero_line():
    high = pd.Series([10.0, 10.0, 10.0, 10.0])
    low = pd.Series([10.0, 10.0, 10.0, 10.0])
    close = pd.Series([10.0, 10.0, 10.0, 10.0])
    volume = pd.Series([100.0, 100.0, 100.0, 100.0])

    result = adl(high, low, close, volume)
    assert (result == 0.0).all()


def test_adl_input_validation():
    with pytest.raises(TypeError):
        adl([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
