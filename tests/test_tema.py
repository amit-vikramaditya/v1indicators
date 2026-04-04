import pandas as pd
import pytest

from v1indicators.overlap import tema


def test_tema_basic():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = tema(close, length=3)

    ema1 = close.ewm(span=3, adjust=False).mean()
    ema2 = ema1.ewm(span=3, adjust=False).mean()
    ema3 = ema2.ewm(span=3, adjust=False).mean()
    expected = 3.0 * ema1 - 3.0 * ema2 + ema3
    expected.name = "TEMA_3"

    pd.testing.assert_series_equal(result, expected)


def test_tema_input_validation():
    with pytest.raises(ValueError):
        tema(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(TypeError):
        tema([1.0, 2.0], length=2)
