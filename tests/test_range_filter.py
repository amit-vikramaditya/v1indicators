"""Tests for the extracted ATR range filter (historical FLOOP core)."""

import numpy as np
import pandas as pd
import pytest

from v1indicators import range_filter
from v1indicators.derived.trend.range_filter import _range_filter_kernel


def test_range_filter_kernel_hand_computed_recursion():
    # Constant band width 1.0; hand-stepped hysteresis recursion.
    src = np.array([100.0, 100.5, 103.0, 100.0, 100.1])
    rng = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    filt, trend, signal = _range_filter_kernel(src, rng)

    assert filt[0] == 100.0 and trend[0] == 0 and signal[0] == 0
    # inside band -> hold
    assert filt[1] == 100.0 and trend[1] == 0 and signal[1] == 0
    # 103 > 100 + 1 -> step up to 103 - 1, trend flips to +1
    assert filt[2] == 102.0 and trend[2] == 1 and signal[2] == 1
    # 100 < 102 - 1 -> step down to 100 + 1, trend flips to -1
    assert filt[3] == 101.0 and trend[3] == -1 and signal[3] == -1
    # inside band -> hold filter and trend, no signal
    assert filt[4] == 101.0 and trend[4] == -1 and signal[4] == 0


def test_range_filter_kernel_nan_carries_state():
    src = np.array([100.0, np.nan, 101.0])
    rng = np.array([1.0, 1.0, 1.0])
    filt, trend, signal = _range_filter_kernel(src, rng)
    # NaN bar freezes the state without emitting anything
    assert np.isnan(filt[1]) or filt[1] == filt[0]
    assert trend[1] == trend[0] and signal[1] == 0
    # recovery continues from the carried filter value
    assert not np.isnan(filt[2])


def test_range_filter_public_output_contract():
    n = 400
    rng = np.random.default_rng(11)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, n)))
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    out = range_filter(high, low, close)
    assert list(out.columns) == ["RANGE_FILTER", "RANGE_FILTER_TREND", "RANGE_FILTER_SIGNAL"]
    assert set(out["RANGE_FILTER_TREND"].unique()).issubset({-1, 0, 1})
    # signals fire only on trend changes
    trend = out["RANGE_FILTER_TREND"].to_numpy()
    signal = out["RANGE_FILTER_SIGNAL"].to_numpy()
    changes = np.zeros(n, dtype=bool)
    changes[1:] = trend[1:] != trend[:-1]
    assert np.array_equal(signal != 0, changes & (trend != 0))


def test_range_filter_validation():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        range_filter(s, s, s, sensitivity=0)
    with pytest.raises(ValueError):
        range_filter(s, s, s, atr_length=0)
    with pytest.raises(ValueError):
        range_filter(s, s, s, atr_multiplier=0.0)
