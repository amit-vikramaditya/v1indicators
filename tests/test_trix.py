import numpy as np
import pandas as pd
import pytest

from v1indicators.momentum import trix


def test_trix_basic():
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 15.0])
    result = trix(close, length=3, drift=1)

    ema1 = close.ewm(span=3, adjust=False, min_periods=3).mean()
    ema2 = ema1.ewm(span=3, adjust=False, min_periods=3).mean()
    ema3 = ema2.ewm(span=3, adjust=False, min_periods=3).mean()
    expected = 100.0 * ema3.pct_change(1)
    expected.name = "TRIX_3_1"

    pd.testing.assert_series_equal(result, expected)


def test_trix_constant_series():
    close = pd.Series([10.0] * 30)
    result = trix(close, length=3)
    # NaN through the nested-EMA warmup (3 stages x (length-1) = bar 6) plus
    # the pct_change lag; constant series is identically zero afterwards.
    assert result.iloc[:7].isna().all()
    assert not np.isnan(result.iloc[7])
    assert (result.iloc[7:] == 0.0).all()


def test_trix_input_validation():
    with pytest.raises(ValueError):
        trix(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(ValueError):
        trix(pd.Series([1.0, 2.0]), length=2, drift=0)

    with pytest.raises(TypeError):
        trix([1.0, 2.0], length=2)
