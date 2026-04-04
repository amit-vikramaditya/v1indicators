import numpy as np
import pandas as pd
import pytest

from v1indicators.trend import ema_rsi_signal


def test_ema_rsi_signal_basic():
    close = pd.Series([10.0, 10.3, 10.6, 10.2, 9.8, 9.5, 9.9, 10.4, 10.8])
    result = ema_rsi_signal(
        close,
        fast_length=3,
        slow_length=5,
        rsi_length=3,
        rsi_buy_level=55.0,
        rsi_sell_level=45.0,
    )

    fast = close.ewm(span=3, adjust=False).mean()
    slow = close.ewm(span=5, adjust=False).mean()

    pd.testing.assert_series_equal(result["EMA_FAST"], fast, check_names=False)
    pd.testing.assert_series_equal(result["EMA_SLOW"], slow, check_names=False)
    assert set(result["TREND"].dropna().unique()).issubset({-1, 0, 1})


def test_ema_rsi_signal_input_validation():
    s = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError):
        ema_rsi_signal(s, fast_length=0)

    with pytest.raises(TypeError):
        ema_rsi_signal([1.0, 2.0])
