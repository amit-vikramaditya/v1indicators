import pandas as pd
import pytest

from v1indicators.overlap.tma import tma


def test_tma_odd_length():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = tma(close, length=5)

    expected = close.rolling(3).mean().rolling(3).mean()
    expected.name = "TMA_5"

    pd.testing.assert_series_equal(result, expected)


def test_tma_even_length():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    result = tma(close, length=6)

    expected = close.rolling(3).mean().rolling(4).mean()
    expected.name = "TMA_6"

    pd.testing.assert_series_equal(result, expected)


def test_tma_input_validation():
    with pytest.raises(ValueError):
        tma(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(TypeError):
        tma([1.0, 2.0], length=2)
