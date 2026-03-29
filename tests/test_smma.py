import pandas as pd
import pytest

from v1indicators.overlap.smma import smma


def test_smma_basic():
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0])
    result = smma(close, length=3)
    expected = close.ewm(alpha=1.0 / 3.0, adjust=False).mean()
    expected.name = "SMMA_3"
    pd.testing.assert_series_equal(result, expected)


def test_smma_input_validation():
    with pytest.raises(ValueError):
        smma(pd.Series([1.0, 2.0]), length=0)
    with pytest.raises(TypeError):
        smma([1.0, 2.0], length=2)
