import pandas as pd
import pytest

from v1indicators.momentum.macd_state import macd_state


def test_macd_state_basic_sma_signal():
    close = pd.Series([10.0, 11.0, 12.0, 11.5, 12.2, 13.0, 12.8, 13.4])

    result = macd_state(close, fast=3, slow=5, signal=2, signal_ma="sma")

    fast_ema = close.ewm(span=3, adjust=False).mean()
    slow_ema = close.ewm(span=5, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.rolling(2).mean()
    hist = macd_line - signal_line

    pd.testing.assert_series_equal(result["MACD"], macd_line, check_names=False)
    pd.testing.assert_series_equal(result["MACD_SIGNAL"], signal_line, check_names=False)
    pd.testing.assert_series_equal(result["MACD_HIST"], hist, check_names=False)
    assert list(result.columns) == [
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "MACD_ABOVE_SIGNAL",
        "MACD_CROSS_UP",
        "MACD_CROSS_DOWN",
        "HIST_UP_POS",
        "HIST_DOWN_POS",
        "HIST_DOWN_NEG",
        "HIST_UP_NEG",
    ]


def test_macd_state_input_validation():
    with pytest.raises(ValueError):
        macd_state(pd.Series([1.0, 2.0]), fast=0)

    with pytest.raises(ValueError):
        macd_state(pd.Series([1.0, 2.0]), signal_ma="invalid")

    with pytest.raises(TypeError):
        macd_state([1.0, 2.0])
