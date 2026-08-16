"""Tests for fractional_difference and the rolling Hurst exponent."""

import numpy as np
import pandas as pd
import pytest

from v1indicators import fractional_difference, hurst


def test_fracdiff_d1_is_first_difference():
    x = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0])
    out = fractional_difference(x, d=1.0)
    assert np.allclose(out.to_numpy()[1:], np.diff(x.to_numpy()), equal_nan=True)
    assert np.isnan(out.iloc[0])


def test_fracdiff_d0_is_identity():
    x = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0])
    out = fractional_difference(x, d=0.0)
    assert np.allclose(out.to_numpy(), x.to_numpy(), equal_nan=True)


def test_fracdiff_hand_computed_values():
    # d=0.5 with min_weight=0.07 truncates to w = [1, -0.5, -0.125]
    x = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0])
    out = fractional_difference(x, d=0.5, min_weight=0.07)
    expected = [
        np.nan,
        np.nan,
        x[2] - 0.5 * x[1] - 0.125 * x[0],
        x[3] - 0.5 * x[2] - 0.125 * x[1],
        x[4] - 0.5 * x[3] - 0.125 * x[2],
    ]
    assert np.allclose(out.to_numpy(), expected, equal_nan=True)
    assert out.name == "FRACDIFF_0.5"


def test_fracdiff_nan_propagation_is_local():
    x = pd.Series([10.0, np.nan, 11.0, 12.0, 13.0])
    out = fractional_difference(x, d=0.5, min_weight=0.07)
    # w has 3 terms: bar i needs x[i], x[i-1], x[i-2] all finite. The NaN at
    # index 1 poisons bars 1..3; bar 4's window (2, 3, 4) is clean.
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])
    assert np.isclose(out.iloc[4], 13.0 - 0.5 * 12.0 - 0.125 * 11.0)


def test_fracdiff_input_validation():
    x = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        fractional_difference(x, d=1.5)
    with pytest.raises(ValueError):
        fractional_difference(x, d=0.5, min_weight=0.0)


def test_hurst_trending_series_is_high():
    ramp = pd.Series(np.linspace(100.0, 200.0, 600))
    h = hurst(ramp, window=100)
    assert np.isnan(h.iloc[:99]).all()
    assert h.iloc[99] == 1.0
    assert (h.iloc[99:] > 0.9).all()


def test_hurst_mean_reverting_series_is_low():
    sine = pd.Series(100.0 + 10.0 * np.sin(np.linspace(0.0, 40.0 * np.pi, 600)))
    h = hurst(sine, window=100)
    # deterministic anti-persistent series; asserted only directionally
    assert h.iloc[-1] < 0.5


def test_hurst_constant_series_is_nan():
    const = pd.Series(np.full(300, 100.0))
    h = hurst(const, window=100)
    assert np.isnan(h.iloc[-1])


def test_hurst_input_validation():
    x = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        hurst(x, window=10)
