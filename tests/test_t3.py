import pandas as pd
import pytest

from v1indicators.overlap.t3 import t3


def test_t3_basic():
    close = pd.Series([10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 14.0])
    result = t3(close, length=3, factor=0.7)

    e1 = close.ewm(span=3, adjust=False).mean()
    e2 = e1.ewm(span=3, adjust=False).mean()
    e3 = e2.ewm(span=3, adjust=False).mean()
    e4 = e3.ewm(span=3, adjust=False).mean()
    e5 = e4.ewm(span=3, adjust=False).mean()
    e6 = e5.ewm(span=3, adjust=False).mean()

    a = 0.7
    c1 = -a**3
    c2 = 3 * a**2 + 3 * a**3
    c3 = -6 * a**2 - 3 * a - 3 * a**3
    c4 = 1 + 3 * a + a**3 + 3 * a**2

    expected = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    expected.name = "T3_3_0.7"

    pd.testing.assert_series_equal(result, expected)


def test_t3_input_validation():
    with pytest.raises(ValueError):
        t3(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(ValueError):
        t3(pd.Series([1.0, 2.0]), length=2, factor=0.0)

    with pytest.raises(TypeError):
        t3([1.0, 2.0], length=2)
