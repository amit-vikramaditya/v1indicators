import pandas as pd
import pytest

from v1indicators.overlap import zlema


def test_zlema_basic():
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0])
    result = zlema(close, length=4)

    lag = int((4 - 1) / 2)
    adjusted = close + (close - close.shift(lag))
    expected = adjusted.ewm(span=4, adjust=False).mean()
    expected.name = "ZLEMA_4"

    pd.testing.assert_series_equal(result, expected)


def test_zlema_length_one_identity():
    close = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = zlema(close, length=1)
    expected = close.copy()
    expected.name = "ZLEMA_1"

    pd.testing.assert_series_equal(result, expected)


def test_zlema_input_validation():
    with pytest.raises(ValueError):
        zlema(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(TypeError):
        zlema([1.0, 2.0], length=2)
