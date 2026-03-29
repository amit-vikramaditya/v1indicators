import numpy as np
import pandas as pd
import pytest

from v1indicators.trend.direction_regime import direction_regime


def test_direction_regime_basic():
    close = pd.Series([10.0, 10.3, 10.7, 10.6, 10.1, 9.8, 9.5, 9.7])
    result = direction_regime(close, ma_length=3, ma_type="sma", threshold_percent=0.2)

    ma = close.rolling(3).mean()
    threshold = close * 0.002
    change = (close - ma).abs()
    up = (close > ma) & (change > threshold)
    down = (close < ma) & (change > threshold)
    trend = pd.Series(np.where(up, 1, np.where(down, -1, 0)), index=close.index, dtype=np.int8)

    pd.testing.assert_series_equal(result["DIRECTION_MA"], ma, check_names=False)
    pd.testing.assert_series_equal(result["TREND"], trend, check_names=False)


def test_direction_regime_input_validation():
    s = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError):
        direction_regime(s, ma_length=0)

    with pytest.raises(ValueError):
        direction_regime(s, threshold_percent=-1.0)

    with pytest.raises(ValueError):
        direction_regime(s, ma_type="invalid")

    with pytest.raises(TypeError):
        direction_regime([1.0, 2.0])
