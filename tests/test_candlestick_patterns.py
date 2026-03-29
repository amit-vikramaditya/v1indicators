import pandas as pd
import pytest

from v1indicators.momentum.candlestick_patterns import candlestick_patterns


def test_candlestick_patterns_basic_shape():
    open_ = pd.Series([10.0, 11.0, 10.5, 10.2, 10.8, 11.3])
    high = pd.Series([10.5, 11.5, 11.0, 10.8, 11.2, 11.6])
    low = pd.Series([9.7, 10.6, 10.1, 9.9, 10.4, 10.9])
    close = pd.Series([10.2, 10.7, 10.9, 10.1, 11.1, 11.0])

    result = candlestick_patterns(open_, high, low, close)

    expected_cols = [
        "DOJI",
        "BULLISH_ENGULFING",
        "BEARISH_ENGULFING",
        "HAMMER",
        "INVERTED_HAMMER",
        "MORNING_STAR",
        "EVENING_STAR",
    ]
    assert list(result.columns) == expected_cols
    assert len(result) == len(close)


def test_candlestick_patterns_detects_doji():
    open_ = pd.Series([10.0, 10.0, 10.0])
    high = pd.Series([10.5, 10.5, 10.5])
    low = pd.Series([9.5, 9.5, 9.5])
    close = pd.Series([10.01, 9.99, 10.0])

    result = candlestick_patterns(open_, high, low, close, doji_size=0.05)
    assert result["DOJI"].all()


def test_candlestick_patterns_input_validation():
    with pytest.raises(ValueError):
        candlestick_patterns(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            doji_size=0.0,
        )

    with pytest.raises(TypeError):
        candlestick_patterns([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
