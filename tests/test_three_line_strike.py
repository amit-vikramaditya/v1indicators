import pandas as pd
import pytest

from v1indicators.momentum.three_line_strike import three_line_strike


def test_three_line_strike_detects_patterns():
    # Bullish strike at index 3.
    open_b = pd.Series([10.0, 9.8, 9.6, 9.5])
    close_b = pd.Series([9.7, 9.5, 9.3, 9.7])
    result_b = three_line_strike(open_b, close_b)
    assert bool(result_b["BULLISH_THREE_LINE_STRIKE"].iloc[3])

    # Bearish strike at index 3.
    open_s = pd.Series([9.0, 9.3, 9.6, 9.7])
    close_s = pd.Series([9.2, 9.5, 9.8, 9.5])
    result_s = three_line_strike(open_s, close_s)
    assert bool(result_s["BEARISH_THREE_LINE_STRIKE"].iloc[3])


def test_three_line_strike_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(TypeError):
        three_line_strike([1.0, 2.0], s)
