import pandas as pd
import pytest

from v1indicators.momentum import candlestick_patterns_extended


def test_candlestick_patterns_extended_columns_and_doji():
    open_ = pd.Series([10.0, 10.0, 10.0, 9.8, 9.7, 9.6])
    high = pd.Series([10.5, 10.4, 10.5, 10.1, 10.0, 9.9])
    low = pd.Series([9.5, 9.6, 9.5, 9.4, 9.3, 9.2])
    close = pd.Series([10.01, 9.99, 10.0, 10.0, 9.8, 9.7])

    result = candlestick_patterns_extended(open_, high, low, close, doji_size=0.05, trend_bars=2)

    expected_cols = [
        "DOJI",
        "BULLISH_HARAMI",
        "BEARISH_HARAMI",
        "BULLISH_ENGULFING",
        "BEARISH_ENGULFING",
        "PIERCING_LINE",
        "BULLISH_BELT_HOLD",
        "BULLISH_KICKER",
        "BEARISH_KICKER",
        "HANGING_MAN",
        "MORNING_STAR",
        "EVENING_STAR",
        "SHOOTING_STAR",
        "HAMMER",
        "INVERTED_HAMMER",
    ]

    assert list(result.columns) == expected_cols
    assert bool(result["DOJI"].iloc[0])
    assert bool(result["DOJI"].iloc[1])
    assert bool(result["DOJI"].iloc[2])


def test_candlestick_patterns_extended_input_validation():
    s = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError):
        candlestick_patterns_extended(s, s, s, s, doji_size=0.0)

    with pytest.raises(ValueError):
        candlestick_patterns_extended(s, s, s, s, trend_bars=0)

    with pytest.raises(TypeError):
        candlestick_patterns_extended([1.0, 2.0], s, s, s)
