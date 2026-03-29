import numpy as np
import pandas as pd
import pytest

from v1indicators.trend.trendline_breaks import trendline_breaks


def _expected_breaks(close, pivot_high, pivot_low, length):
    n = close.size
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    slope_upper = np.zeros(n, dtype=np.float64)
    slope_lower = np.zeros(n, dtype=np.float64)
    break_up = np.zeros(n, dtype=bool)
    break_down = np.zeros(n, dtype=bool)

    cur_upper = np.nan
    cur_lower = np.nan
    prev_dyn_upper = np.nan
    prev_dyn_lower = np.nan

    for i in range(n):
        if not np.isnan(pivot_high[i]):
            cur_upper = pivot_high[i]
        elif not np.isnan(cur_upper):
            cur_upper = cur_upper

        if not np.isnan(pivot_low[i]):
            cur_lower = pivot_low[i]
        elif not np.isnan(cur_lower):
            cur_lower = cur_lower

        upper[i] = cur_upper
        lower[i] = cur_lower
        slope_upper[i] = 0.0
        slope_lower[i] = 0.0

        dyn_upper = cur_upper
        dyn_lower = cur_lower

        if i > 0:
            if (
                not np.isnan(dyn_upper)
                and not np.isnan(prev_dyn_upper)
                and close[i] > dyn_upper
                and close[i - 1] <= prev_dyn_upper
            ):
                break_up[i] = True
            if (
                not np.isnan(dyn_lower)
                and not np.isnan(prev_dyn_lower)
                and close[i] < dyn_lower
                and close[i - 1] >= prev_dyn_lower
            ):
                break_down[i] = True

        prev_dyn_upper = dyn_upper
        prev_dyn_lower = dyn_lower

    return upper, lower, slope_upper, slope_lower, break_up, break_down


def test_trendline_breaks_basic():
    high = pd.Series([10.0, 11.0, 12.0, 11.0, 10.5, 11.5, 12.5, 11.8, 11.0])
    low = pd.Series([9.0, 9.5, 10.0, 9.6, 9.3, 10.0, 10.8, 10.2, 9.8])
    close = pd.Series([9.4, 10.5, 11.8, 10.1, 9.7, 10.8, 12.2, 10.4, 9.9])

    result = trendline_breaks(high, low, close, length=2, mult=0.0, slope_method="atr")

    window = 5
    pivot_high = high.where(high == high.rolling(window).max().shift(-2)).to_numpy(dtype=np.float64)
    pivot_low = low.where(low == low.rolling(window).min().shift(-2)).to_numpy(dtype=np.float64)
    expected = _expected_breaks(close.to_numpy(dtype=np.float64), pivot_high, pivot_low, length=2)

    np.testing.assert_allclose(result["TRENDLINE_UPPER"], expected[0], equal_nan=True)
    np.testing.assert_allclose(result["TRENDLINE_LOWER"], expected[1], equal_nan=True)
    np.testing.assert_allclose(result["TRENDLINE_SLOPE_UPPER"], expected[2], equal_nan=True)
    np.testing.assert_allclose(result["TRENDLINE_SLOPE_LOWER"], expected[3], equal_nan=True)
    np.testing.assert_array_equal(result["BREAKOUT_UP"].to_numpy(), expected[4])
    np.testing.assert_array_equal(result["BREAKOUT_DOWN"].to_numpy(), expected[5])


def test_trendline_breaks_input_validation():
    s = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        trendline_breaks(s, s, s, length=1)

    with pytest.raises(ValueError):
        trendline_breaks(s, s, s, slope_method="unknown")

    with pytest.raises(TypeError):
        trendline_breaks([1.0, 2.0], s, s)
