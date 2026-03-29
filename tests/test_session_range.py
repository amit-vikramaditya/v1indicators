import pandas as pd
import pytest

from v1indicators.trend.session_range import session_range


def test_session_range_basic():
    idx = pd.date_range("2026-01-01 08:00", periods=8, freq="30min")
    high = pd.Series([10.0, 10.2, 10.5, 10.4, 10.8, 10.7, 10.6, 10.9], index=idx)
    low = pd.Series([9.5, 9.7, 10.0, 9.9, 10.1, 10.2, 10.0, 10.3], index=idx)
    close = pd.Series([9.8, 10.0, 10.3, 10.2, 10.6, 10.5, 10.1, 10.8], index=idx)

    result = session_range(high, low, close, start="08:30", end="10:30")
    assert "SESSION_ACTIVE" in result.columns
    assert "SESSION_RANGE" in result.columns
    assert len(result) == len(idx)


def test_session_range_requires_datetime_index():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        session_range(s, s, s)
