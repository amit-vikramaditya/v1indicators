import pandas as pd
import pytest

from v1indicators.volume import cmf


def test_cmf_basic():
    high = pd.Series([12.0, 13.0, 14.0, 15.0, 16.0])
    low = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0])
    close = pd.Series([10.0, 12.0, 13.0, 14.0, 15.0])
    volume = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0])

    result = cmf(high, low, close, volume, length=3)

    mfm = ((close - low) - (high - close)) / (high - low)
    mfv = mfm * volume
    expected = mfv.rolling(3).sum() / volume.rolling(3).sum()
    expected.name = "CMF_3"

    pd.testing.assert_series_equal(result, expected)


def test_cmf_zero_range_nan():
    high = pd.Series([10.0, 10.0, 10.0, 10.0])
    low = pd.Series([10.0, 10.0, 10.0, 10.0])
    close = pd.Series([10.0, 10.0, 10.0, 10.0])
    volume = pd.Series([100.0, 100.0, 100.0, 100.0])

    result = cmf(high, low, close, volume, length=2)
    assert result.isna().all()


def test_cmf_input_validation():
    with pytest.raises(ValueError):
        cmf(
            pd.Series([1.0]),
            pd.Series([1.0]),
            pd.Series([1.0]),
            pd.Series([1.0]),
            length=0,
        )

    with pytest.raises(TypeError):
        cmf([1.0], pd.Series([1.0]), pd.Series([1.0]), pd.Series([1.0]), length=2)
