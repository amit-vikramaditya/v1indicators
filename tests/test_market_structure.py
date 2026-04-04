import numpy as np
import pandas as pd
import pytest

from v1indicators.trend import market_structure


def _expected_structure(close, resistance, support):
    n = close.size
    trend = np.zeros(n, dtype=np.int8)
    bullish_bos = np.zeros(n, dtype=bool)
    bearish_bos = np.zeros(n, dtype=bool)
    bullish_choch = np.zeros(n, dtype=bool)
    bearish_choch = np.zeros(n, dtype=bool)

    cur = 0
    for i in range(1, n):
        cross_up = (
            not np.isnan(resistance[i - 1])
            and close[i - 1] <= resistance[i - 1]
            and close[i] > resistance[i - 1]
        )
        cross_down = (
            not np.isnan(support[i - 1])
            and close[i - 1] >= support[i - 1]
            and close[i] < support[i - 1]
        )

        if cross_up:
            if cur == -1:
                bullish_choch[i] = True
            else:
                bullish_bos[i] = True
            cur = 1
        elif cross_down:
            if cur == 1:
                bearish_choch[i] = True
            else:
                bearish_bos[i] = True
            cur = -1

        trend[i] = cur

    return trend, bullish_bos, bearish_bos, bullish_choch, bearish_choch


def test_market_structure_basic():
    high = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 12.0])
    low = pd.Series([9.0, 10.0, 11.0, 10.0, 11.0, 10.0, 12.0, 11.0, 10.5])
    close = pd.Series([9.5, 10.8, 11.7, 10.2, 12.6, 10.1, 13.7, 10.9, 10.2])

    result = market_structure(high, low, close, left=1, right=1)

    window = 3
    pivot_high = high.where(high == high.rolling(window).max().shift(-1))
    pivot_low = low.where(low == low.rolling(window).min().shift(-1))
    resistance = pivot_high.ffill().to_numpy(dtype=np.float64)
    support = pivot_low.ffill().to_numpy(dtype=np.float64)

    expected = _expected_structure(close.to_numpy(dtype=np.float64), resistance, support)

    pd.testing.assert_series_equal(result["SWING_HIGH"], pivot_high, check_names=False)
    pd.testing.assert_series_equal(result["SWING_LOW"], pivot_low, check_names=False)
    np.testing.assert_array_equal(result["MARKET_TREND"].to_numpy(), expected[0])
    np.testing.assert_array_equal(result["BULLISH_BOS"].to_numpy(), expected[1])
    np.testing.assert_array_equal(result["BEARISH_BOS"].to_numpy(), expected[2])
    np.testing.assert_array_equal(result["BULLISH_CHOCH"].to_numpy(), expected[3])
    np.testing.assert_array_equal(result["BEARISH_CHOCH"].to_numpy(), expected[4])


def test_market_structure_input_validation():
    s = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        market_structure(s, s, s, left=0, right=1)

    with pytest.raises(TypeError):
        market_structure([1.0, 2.0], s, s)
