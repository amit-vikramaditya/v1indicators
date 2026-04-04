import numpy as np
import pandas as pd
import pytest

from v1indicators.overlap import kama


def _kama_expected(close: pd.Series, length: int, fast: int, slow: int) -> pd.Series:
    change = (close - close.shift(length)).abs()
    volatility = close.diff().abs().rolling(length).sum().replace(0.0, np.nan)
    er = change / volatility

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    out = np.full(close.shape[0], np.nan, dtype=np.float64)
    if close.shape[0] >= length:
        out[length - 1] = close.iloc[:length].mean()
        for i in range(length, close.shape[0]):
            if np.isnan(out[i - 1]) or np.isnan(close.iloc[i]) or np.isnan(sc.iloc[i]):
                out[i] = np.nan
            else:
                out[i] = out[i - 1] + sc.iloc[i] * (close.iloc[i] - out[i - 1])

    return pd.Series(out, index=close.index, name=f"KAMA_{length}_{fast}_{slow}")


def test_kama_basic():
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0, 15.0, 16.0, 15.0])
    result = kama(close, length=4, fast=2, slow=10)
    expected = _kama_expected(close, length=4, fast=2, slow=10)

    assert np.allclose(result.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert result.name == expected.name


def test_kama_short_series_all_nan():
    close = pd.Series([1.0, 2.0, 3.0])
    result = kama(close, length=5)
    assert result.isna().all()


def test_kama_input_validation():
    with pytest.raises(ValueError):
        kama(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(ValueError):
        kama(pd.Series([1.0, 2.0]), length=2, fast=0, slow=10)

    with pytest.raises(TypeError):
        kama([1.0, 2.0], length=2)
