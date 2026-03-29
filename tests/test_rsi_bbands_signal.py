import pandas as pd
import pytest

from v1indicators.momentum.rsi_bbands_signal import rsi_bbands_signal


def test_rsi_bbands_signal_basic_columns():
    close = pd.Series([10.0, 9.8, 9.5, 9.3, 9.6, 9.9, 10.2, 10.0, 9.7, 9.4, 9.8, 10.1])
    result = rsi_bbands_signal(
        close,
        rsi_length=3,
        bb_length=3,
        bb_mult=2.0,
    )

    assert list(result.columns) == [
        "RSI",
        "BB_BASIS",
        "BB_UPPER",
        "BB_LOWER",
        "RSI_CROSS_UP",
        "RSI_CROSS_DOWN",
        "PRICE_CROSS_BB_LOWER",
        "PRICE_CROSS_BB_UPPER",
        "LONG_SIGNAL",
        "SHORT_SIGNAL",
    ]
    assert result["LONG_SIGNAL"].dtype == bool
    assert result["SHORT_SIGNAL"].dtype == bool


def test_rsi_bbands_signal_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        rsi_bbands_signal(s, rsi_length=0)
    with pytest.raises(ValueError):
        rsi_bbands_signal(s, bb_mult=0.0)
    with pytest.raises(TypeError):
        rsi_bbands_signal([1.0, 2.0])
