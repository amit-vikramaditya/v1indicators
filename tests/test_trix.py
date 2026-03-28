import numpy as np
import pandas as pd
import pytest

from v1indicators.momentum.trix import trix


def test_trix_basic():
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 15.0])
    result = trix(close, length=3, drift=1)

    ema1 = close.ewm(span=3, adjust=False).mean()
    ema2 = ema1.ewm(span=3, adjust=False).mean()
    ema3 = ema2.ewm(span=3, adjust=False).mean()
    expected = 100.0 * ema3.pct_change(1)
    expected.name = "TRIX_3_1"

    pd.testing.assert_series_equal(result, expected)


def test_trix_constant_series():
    close = pd.Series([10.0] * 10)
    result = trix(close, length=3)
    assert np.isnan(result.iloc[0])
    assert (result.iloc[1:] == 0.0).all()


def test_trix_input_validation():
    with pytest.raises(ValueError):
        trix(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(ValueError):
        trix(pd.Series([1.0, 2.0]), length=2, drift=0)

    with pytest.raises(TypeError):
        trix([1.0, 2.0], length=2)
