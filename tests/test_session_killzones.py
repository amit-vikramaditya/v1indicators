import pandas as pd
import pytest

from v1indicators.trend import session_killzones


def test_session_killzones_basic_columns():
    idx = pd.date_range("2026-01-01 08:00", periods=8, freq="30min")
    high = pd.Series([10.0, 10.2, 10.4, 10.3, 10.7, 10.8, 10.6, 10.9], index=idx)
    low = pd.Series([9.7, 9.8, 10.0, 9.9, 10.2, 10.3, 10.1, 10.4], index=idx)
    close = pd.Series([9.9, 10.0, 10.3, 10.1, 10.5, 10.7, 10.2, 10.8], index=idx)

    result = session_killzones(
        high,
        low,
        close,
        asia=("20:00", "00:00"),
        london=("08:00", "10:00"),
        ny_am=("09:30", "11:00"),
    )

    assert "LONDON_ACTIVE" in result.columns
    assert "NY_AM_RANGE" in result.columns
    assert len(result) == len(idx)


def test_session_killzones_requires_datetime_index():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        session_killzones(s, s, s)
