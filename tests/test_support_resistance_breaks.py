import pandas as pd
import pytest

from v1indicators.trend.support_resistance_breaks import support_resistance_breaks


def test_support_resistance_breaks_detects_breaks():
    open_ = pd.Series([9.5, 10.5, 9.5, 12.0, 7.0])
    high = pd.Series([10.0, 12.0, 9.0, 13.0, 7.2])
    low = pd.Series([8.0, 9.0, 7.0, 11.0, 6.0])
    close = pd.Series([9.0, 10.0, 9.0, 13.0, 6.5])
    volume = pd.Series([100.0, 100.0, 100.0, 1000.0, 1200.0])

    result = support_resistance_breaks(
        open_,
        high,
        low,
        close,
        volume,
        left=1,
        right=1,
        volume_fast=1,
        volume_slow=2,
        volume_threshold=5.0,
    )

    assert bool(result["BREAK_RESISTANCE"].iloc[3])
    assert bool(result["BREAK_SUPPORT"].iloc[4])
    assert result["BREAK_RESISTANCE"].dtype == bool
    assert result["BREAK_SUPPORT"].dtype == bool


def test_support_resistance_breaks_input_validation():
    s = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        support_resistance_breaks(s, s, s, s, s, left=0)

    with pytest.raises(ValueError):
        support_resistance_breaks(s, s, s, s, s, volume_fast=0)

    with pytest.raises(TypeError):
        support_resistance_breaks([1.0, 2.0], s, s, s, s)
