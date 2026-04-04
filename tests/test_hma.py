import numpy as np
import pandas as pd
import pytest

from v1indicators.overlap import hma


def _wma_expected(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=np.float64)
    denom = weights.sum()
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / denom, raw=True)


def test_hma_basic():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = hma(close, length=4)

    half = _wma_expected(close, 2)
    full = _wma_expected(close, 4)
    raw = 2.0 * half - full
    expected = _wma_expected(raw, 2)
    expected.name = "HMA_4"

    pd.testing.assert_series_equal(result, expected)


def test_hma_short_series_all_nan():
    close = pd.Series([1.0, 2.0, 3.0])
    result = hma(close, length=5)
    assert result.isna().all()


def test_hma_input_validation():
    with pytest.raises(ValueError):
        hma(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(TypeError):
        hma([1.0, 2.0, 3.0], length=3)
