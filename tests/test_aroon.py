import numpy as np
import pandas as pd
import pytest

from v1indicators.trend import aroon, aroon_down, aroon_osc, aroon_up


def _aroon_up_expected(high: pd.Series, length: int) -> pd.Series:
    def calc(window):
        periods_since_high = np.argmax(window[::-1])
        return 100.0 * (length - periods_since_high) / length

    return high.rolling(length).apply(calc, raw=True)


def _aroon_down_expected(low: pd.Series, length: int) -> pd.Series:
    def calc(window):
        periods_since_low = np.argmin(window[::-1])
        return 100.0 * (length - periods_since_low) / length

    return low.rolling(length).apply(calc, raw=True)


def test_aroon_components_and_oscillator():
    high = pd.Series([10.0, 11.0, 9.0, 12.0, 11.0, 13.0, 12.0])
    low = pd.Series([7.0, 8.0, 6.0, 7.0, 6.0, 8.0, 7.0])

    up = aroon_up(high, length=4)
    down = aroon_down(low, length=4)
    osc = aroon_osc(high, low, length=4)
    primary = aroon(high, low, length=4)

    expected_up = _aroon_up_expected(high, 4)
    expected_up.name = "AROON_UP_4"
    expected_down = _aroon_down_expected(low, 4)
    expected_down.name = "AROON_DOWN_4"
    expected_osc = expected_up - expected_down
    expected_osc.name = "AROON_OSC_4"

    pd.testing.assert_series_equal(up, expected_up)
    pd.testing.assert_series_equal(down, expected_down)
    pd.testing.assert_series_equal(osc, expected_osc)
    pd.testing.assert_series_equal(primary, expected_osc)


def test_aroon_short_series_all_nan():
    high = pd.Series([1.0, 2.0])
    low = pd.Series([1.0, 0.5])

    assert aroon_up(high, length=5).isna().all()
    assert aroon_down(low, length=5).isna().all()
    assert aroon(high, low, length=5).isna().all()


def test_aroon_input_validation():
    with pytest.raises(ValueError):
        aroon_up(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(ValueError):
        aroon_down(pd.Series([1.0, 2.0]), length=0)

    with pytest.raises(TypeError):
        aroon_up([1.0, 2.0], length=2)
