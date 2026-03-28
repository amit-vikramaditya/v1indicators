import pandas as pd
import pytest

from v1indicators.momentum.ultimate_oscillator import ultimate_oscillator


def test_ultimate_oscillator_basic():
    high = pd.Series([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 16.0, 18.0])
    low = pd.Series([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 14.0])
    close = pd.Series([10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 15.0, 17.0])

    result = ultimate_oscillator(high, low, close, short=2, medium=3, long=4)

    prev_close = close.shift(1)
    min_lp = pd.concat([low, prev_close], axis=1).min(axis=1)
    max_hp = pd.concat([high, prev_close], axis=1).max(axis=1)
    bp = close - min_lp
    tr = max_hp - min_lp

    avg_short = bp.rolling(2).sum() / tr.rolling(2).sum()
    avg_medium = bp.rolling(3).sum() / tr.rolling(3).sum()
    avg_long = bp.rolling(4).sum() / tr.rolling(4).sum()
    expected = 100.0 * (4.0 * avg_short + 2.0 * avg_medium + 1.0 * avg_long) / 7.0
    expected.name = "UO_2_3_4"

    pd.testing.assert_series_equal(result, expected)


def test_ultimate_oscillator_flat_range_nan():
    high = pd.Series([10.0] * 8)
    low = pd.Series([10.0] * 8)
    close = pd.Series([10.0] * 8)

    result = ultimate_oscillator(high, low, close, short=2, medium=3, long=4)
    assert result.isna().all()


def test_ultimate_oscillator_input_validation():
    with pytest.raises(ValueError):
        ultimate_oscillator(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            short=0,
            medium=2,
            long=3,
        )

    with pytest.raises(ValueError):
        ultimate_oscillator(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            short=3,
            medium=2,
            long=4,
        )

    with pytest.raises(TypeError):
        ultimate_oscillator([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
