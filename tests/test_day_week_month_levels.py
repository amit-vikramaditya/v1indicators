import pandas as pd
import pytest

from v1indicators.trend import day_week_month_levels


def test_day_week_month_levels_basic():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    open_ = pd.Series([10, 11, 12, 13, 12, 11, 12, 13, 14, 15], index=idx, dtype=float)
    high = open_ + 1.0
    low = open_ - 1.0

    result = day_week_month_levels(open_, high, low)
    assert list(result.columns) == [
        "PD_OPEN",
        "PD_HIGH",
        "PD_LOW",
        "PW_OPEN",
        "PW_HIGH",
        "PW_LOW",
        "PM_OPEN",
        "PM_HIGH",
        "PM_LOW",
    ]
    assert len(result) == len(idx)


def test_day_week_month_levels_requires_datetime_index():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        day_week_month_levels(s, s, s)
