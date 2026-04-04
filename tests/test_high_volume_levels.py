import pandas as pd
import pytest

from v1indicators.trend import high_volume_levels


def test_high_volume_levels_basic_columns():
    open_ = pd.Series([10.0, 11.0, 10.5, 10.8, 10.2, 11.4, 10.1])
    high = pd.Series([10.5, 11.5, 11.0, 11.2, 10.7, 11.8, 10.6])
    low = pd.Series([9.6, 10.7, 10.0, 10.1, 9.8, 10.9, 9.7])
    close = pd.Series([10.2, 10.9, 10.3, 10.7, 10.1, 11.6, 9.9])
    volume = pd.Series([100.0, 140.0, 90.0, 110.0, 85.0, 180.0, 170.0])

    result = high_volume_levels(open_, high, low, close, volume, lookback=1, vol_length=2)
    assert list(result.columns) == [
        "HV_PIVOT_HIGH",
        "HV_PIVOT_LOW",
        "HV_RESISTANCE",
        "HV_SUPPORT",
        "HV_BREAK_RESISTANCE",
        "HV_BREAK_SUPPORT",
    ]
    assert result["HV_BREAK_RESISTANCE"].dtype == bool
    assert result["HV_BREAK_SUPPORT"].dtype == bool


def test_high_volume_levels_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        high_volume_levels(s, s, s, s, s, lookback=0)
    with pytest.raises(TypeError):
        high_volume_levels([1.0, 2.0], s, s, s, s)
