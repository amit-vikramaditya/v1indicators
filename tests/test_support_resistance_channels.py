import pandas as pd
import pytest

from v1indicators.trend import support_resistance_channels


def test_support_resistance_channels_basic_shape():
    high = pd.Series([10.0, 11.5, 11.0, 12.0, 11.8, 12.2, 11.6, 12.4, 11.9, 12.6])
    low = pd.Series([9.2, 10.0, 9.8, 10.4, 10.5, 10.8, 10.2, 11.0, 10.6, 11.3])
    close = pd.Series([9.8, 11.0, 10.3, 11.6, 11.2, 11.9, 10.9, 12.0, 11.0, 12.4])

    result = support_resistance_channels(
        high,
        low,
        close,
        pivot_period=1,
        channel_width_pct=8.0,
        min_strength=1,
        loopback=8,
    )

    assert list(result.columns) == [
        "PIVOT_VALUE",
        "SR_RESISTANCE",
        "SR_SUPPORT",
        "SR_RESISTANCE_STRENGTH",
        "SR_SUPPORT_STRENGTH",
        "BREAK_RESISTANCE",
        "BREAK_SUPPORT",
    ]
    assert len(result) == len(close)


def test_support_resistance_channels_input_validation():
    s = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        support_resistance_channels(s, s, s, pivot_period=0)

    with pytest.raises(ValueError):
        support_resistance_channels(s, s, s, channel_width_pct=0.0)

    with pytest.raises(TypeError):
        support_resistance_channels([1.0, 2.0], s, s)
