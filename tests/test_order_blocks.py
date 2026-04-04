import pandas as pd
import pytest

from v1indicators.trend import order_blocks


def test_order_blocks_detects_bullish_and_bearish():
    open_ = pd.Series([10.0, 9.0, 10.0, 10.0, 8.0, 9.0, 8.0, 8.0])
    close = pd.Series([9.0, 10.0, 11.0, 10.0, 9.0, 8.0, 7.0, 8.0])
    high = pd.Series([11.0, 11.0, 12.0, 11.0, 10.0, 10.0, 9.0, 9.0])
    low = pd.Series([8.0, 8.0, 9.0, 9.0, 7.0, 7.0, 6.0, 7.0])

    result = order_blocks(open_, high, low, close, periods=2, threshold=0.0)

    # Bullish OB expected at index 3 from bearish candle at index 0.
    assert bool(result["BULLISH_OB"].iloc[3])
    assert result["BULLISH_OB_HIGH"].iloc[3] == open_.iloc[0]
    assert result["BULLISH_OB_LOW"].iloc[3] == low.iloc[0]

    # Bearish OB expected at index 7 from bullish candle at index 4.
    assert bool(result["BEARISH_OB"].iloc[7])
    assert result["BEARISH_OB_HIGH"].iloc[7] == high.iloc[4]
    assert result["BEARISH_OB_LOW"].iloc[7] == open_.iloc[4]


def test_order_blocks_use_wicks_changes_levels():
    open_ = pd.Series([10.0, 9.0, 10.0, 10.0, 8.0, 9.0, 8.0, 8.0])
    close = pd.Series([9.0, 10.0, 11.0, 10.0, 9.0, 8.0, 7.0, 8.0])
    high = pd.Series([11.0, 11.0, 12.0, 11.0, 10.0, 10.0, 9.0, 9.0])
    low = pd.Series([8.0, 8.0, 9.0, 9.0, 7.0, 7.0, 6.0, 7.0])

    result = order_blocks(open_, high, low, close, periods=2, threshold=0.0, use_wicks=True)

    assert result["BULLISH_OB_HIGH"].iloc[3] == high.iloc[0]
    assert result["BEARISH_OB_LOW"].iloc[7] == low.iloc[4]


def test_order_blocks_input_validation():
    s = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        order_blocks(s, s, s, s, periods=0)

    with pytest.raises(ValueError):
        order_blocks(s, s, s, s, threshold=-0.1)

    with pytest.raises(TypeError):
        order_blocks([1.0, 2.0], s, s, s)
