"""Tests for the range-based volatility estimators and candle_direction."""

import numpy as np
import pandas as pd

from v1indicators import candle_direction, garman_klass, parkinson


def test_parkinson_known_value():
    # ln(2) = 0.693147...; var = ln(2)^2 / (4 ln 2) = ln(2)/4; sigma = sqrt(ln2/4)
    high = pd.Series([2.0])
    low = pd.Series([1.0])
    out = parkinson(high, low)
    expected = np.sqrt(np.log(2.0) / 4.0)
    assert np.isclose(out.iloc[0], expected)
    assert out.name == "PARKINSON"


def test_parkinson_zero_range():
    high = pd.Series([5.0, 5.0])
    low = pd.Series([5.0, 5.0])
    out = parkinson(high, low)
    assert (out == 0.0).all()


def test_parkinson_window_smoothing():
    high = pd.Series([2.0, 4.0])
    low = pd.Series([1.0, 1.0])
    out = parkinson(high, low, window=2)
    v1 = np.log(2.0) ** 2 / (4.0 * np.log(2.0))
    v2 = np.log(4.0) ** 2 / (4.0 * np.log(2.0))
    assert np.isnan(out.iloc[0])
    assert np.isclose(out.iloc[1], np.sqrt((v1 + v2) / 2.0))
    assert out.name == "PARKINSON_2"


def test_parkinson_input_validation():
    s = pd.Series([1.0, 2.0])
    try:
        parkinson(s, s, window=0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_garman_klass_known_value():
    # O=C=1, H=2, L=1: var = 0.5*ln(2)^2 - (2ln2-1)*0
    open_ = pd.Series([1.0])
    high = pd.Series([2.0])
    low = pd.Series([1.0])
    close = pd.Series([1.0])
    out = garman_klass(open_, high, low, close)
    expected = np.sqrt(0.5 * np.log(2.0) ** 2)
    assert np.isclose(out.iloc[0], expected)
    assert out.name == "GARMAN_KLASS"


def test_garman_klass_invalid_ohlc_returns_nan():
    # For price-consistent OHLC the per-bar estimate is always >= 0; only
    # inconsistent inputs (close outside [low, high]) can drive it negative,
    # and those must surface as NaN, never a clipped zero.
    open_ = pd.Series([1.0])
    high = pd.Series([1.1])
    low = pd.Series([1.0])
    close = pd.Series([2.0])  # close > high: inconsistent
    out = garman_klass(open_, high, low, close)
    assert np.isnan(out.iloc[0])


def test_garman_klass_window_smoothing():
    open_ = pd.Series([1.0, 1.0])
    high = pd.Series([2.0, 4.0])
    low = pd.Series([1.0, 1.0])
    close = pd.Series([1.0, 1.0])
    out = garman_klass(open_, high, low, close, window=2)
    v1 = 0.5 * np.log(2.0) ** 2
    v2 = 0.5 * np.log(4.0) ** 2
    assert np.isnan(out.iloc[0])
    assert np.isclose(out.iloc[1], np.sqrt((v1 + v2) / 2.0))
    assert out.name == "GARMAN_KLASS_2"


def test_candle_direction_basic():
    open_ = pd.Series([10.0, 10.0, 10.0, 11.0])
    close = pd.Series([11.0, 9.0, 10.0, 12.0])
    out = candle_direction(open_, close)
    assert out.dtype == np.int8
    assert list(out.to_numpy()) == [1, -1, 0, 1]
    assert out.name == "CANDLE_DIR"
