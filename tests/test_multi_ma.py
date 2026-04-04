import pandas as pd
import pytest

from v1indicators.overlap import multi_ma


def test_multi_ma_basic_with_ema():
    close = pd.Series([10.0, 11.0, 12.0, 11.5, 12.5, 13.0, 12.8])
    result = multi_ma(close, length1=3, ma_type1="ema", length2=5, ma_type2="ema")

    fast = close.ewm(span=3, adjust=False).mean()
    slow = close.ewm(span=5, adjust=False).mean()

    pd.testing.assert_series_equal(result["MA_FAST"], fast, check_names=False)
    pd.testing.assert_series_equal(result["MA_SLOW"], slow, check_names=False)
    assert set(result["MA_TREND"].dropna().unique()).issubset({-1, 0, 1})


def test_multi_ma_vwma_requires_volume():
    close = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        multi_ma(close, ma_type1="vwma")


def test_multi_ma_input_validation():
    close = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        multi_ma(close, length1=0)
    with pytest.raises(ValueError):
        multi_ma(close, ma_type1="bad")
    with pytest.raises(TypeError):
        multi_ma([1.0, 2.0])
