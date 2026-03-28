import pandas as pd
import pytest

from v1indicators.overlap.dema import dema


def test_dema_basic():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = dema(close, length=3)

    ema1 = close.ewm(span=3, adjust=False).mean()
    ema2 = ema1.ewm(span=3, adjust=False).mean()
    expected = 2.0 * ema1 - ema2
    expected.name = "DEMA_3"

    pd.testing.assert_series_equal(result, expected)


def test_dema_input_validation():
    with pytest.raises(ValueError):
        dema(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(TypeError):
        dema([1.0, 2.0], length=2)
