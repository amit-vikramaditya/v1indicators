import pandas as pd
import pytest

from v1indicators.trend import fair_value_gaps


def test_fair_value_gaps_detects_bullish_and_bearish():
    high = pd.Series([10.0, 10.2, 10.6, 10.4, 9.1, 9.0])
    low = pd.Series([9.5, 9.8, 10.5, 9.2, 8.9, 8.7])
    close = pd.Series([9.8, 10.1, 10.55, 9.3, 8.95, 8.8])

    result = fair_value_gaps(high, low, close)

    assert bool(result["BULLISH_FVG"].iloc[2])
    assert bool(result["BEARISH_FVG"].iloc[4])


def test_fair_value_gaps_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        fair_value_gaps(s, s, s, threshold=-1.0)
    with pytest.raises(TypeError):
        fair_value_gaps([1.0, 2.0], s, s)
