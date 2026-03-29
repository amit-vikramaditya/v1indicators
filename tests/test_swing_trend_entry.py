import numpy as np
import pandas as pd
import pytest

from v1indicators.trend.swing_trend_entry import swing_trend_entry


def _expected_gap_source(close: pd.Series, gap_fraction: float) -> pd.Series:
    out = np.full(close.size, np.nan, dtype=np.float64)
    vals = close.to_numpy(dtype=np.float64)
    for i in range(close.size):
        gap = int(i * gap_fraction)
        idx = i - gap
        if idx >= 0:
            out[i] = vals[idx]
    return pd.Series(out, index=close.index, name="GAP_SOURCE")


def test_swing_trend_entry_basic():
    close = pd.Series([10.0, 10.2, 10.5, 10.3, 10.8, 11.0, 10.9, 11.2])
    result = swing_trend_entry(
        close,
        ma_length=3,
        long_ma_length=4,
        time_gap_percent=25.0,
        threshold_percent=0.1,
        ma_type="ema",
    )

    gap_source = _expected_gap_source(close, 0.25)
    pd.testing.assert_series_equal(result["GAP_SOURCE"], gap_source)
    assert list(result.columns) == [
        "GAP_SOURCE",
        "SWING_MA",
        "SWING_LONG_MA",
        "BULLISH",
        "BEARISH",
        "SIDEWAYS",
        "TREND",
        "TREND_CHANGE",
        "TOUCH_MA",
    ]


def test_swing_trend_entry_input_validation():
    s = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        swing_trend_entry(s, ma_length=0)

    with pytest.raises(ValueError):
        swing_trend_entry(s, time_gap_percent=-1.0)

    with pytest.raises(ValueError):
        swing_trend_entry(s, threshold_percent=-1.0)

    with pytest.raises(ValueError):
        swing_trend_entry(s, ma_type="invalid")

    with pytest.raises(TypeError):
        swing_trend_entry([1.0, 2.0], ma_length=2)
