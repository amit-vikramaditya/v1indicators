import pandas as pd
import pytest

from v1indicators.overlap import vwma


def test_vwma_basic():
    close = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    volume = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = vwma(close, volume, length=3)

    expected = (close * volume).rolling(3).sum() / volume.rolling(3).sum()
    expected.name = "VWMA_3"

    pd.testing.assert_series_equal(result, expected)


def test_vwma_zero_volume_denominator():
    close = pd.Series([10.0, 11.0, 12.0, 13.0])
    volume = pd.Series([0.0, 0.0, 0.0, 0.0])

    result = vwma(close, volume, length=2)
    assert result.isna().all()


def test_vwma_input_validation():
    with pytest.raises(ValueError):
        vwma(pd.Series([1.0]), pd.Series([1.0]), length=0)

    with pytest.raises(TypeError):
        vwma([1.0, 2.0], pd.Series([1.0, 2.0]), length=2)

    with pytest.raises(TypeError):
        vwma(pd.Series([1.0, 2.0]), [1.0, 2.0], length=2)
