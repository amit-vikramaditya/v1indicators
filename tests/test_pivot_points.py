import pandas as pd
import pytest

from v1indicators.levels.pivot_points import pivot_points


def test_pivot_points_classic():
    high = pd.Series([10.0, 12.0, 14.0, 13.0])
    low = pd.Series([8.0, 9.0, 11.0, 10.0])
    close = pd.Series([9.0, 11.0, 12.0, 11.5])

    result = pivot_points(high, low, close, method="classic")

    ph = high.shift(1)
    pl = low.shift(1)
    pc = close.shift(1)
    p = (ph + pl + pc) / 3.0
    r1 = 2.0 * p - pl
    s1 = 2.0 * p - ph

    pd.testing.assert_series_equal(result["PIVOT_P"], p, check_names=False)
    pd.testing.assert_series_equal(result["PIVOT_R1"], r1, check_names=False)
    pd.testing.assert_series_equal(result["PIVOT_S1"], s1, check_names=False)


def test_pivot_points_camarilla_has_r4_s4():
    high = pd.Series([10.0, 12.0, 14.0, 13.0])
    low = pd.Series([8.0, 9.0, 11.0, 10.0])
    close = pd.Series([9.0, 11.0, 12.0, 11.5])

    result = pivot_points(high, low, close, method="camarilla")
    assert result["PIVOT_R4"].notna().sum() > 0
    assert result["PIVOT_S4"].notna().sum() > 0


def test_pivot_points_input_validation():
    with pytest.raises(ValueError):
        pivot_points(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            method="unknown",
        )

    with pytest.raises(TypeError):
        pivot_points([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
