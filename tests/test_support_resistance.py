import pandas as pd
import pytest

from v1indicators.levels.support_resistance import support_resistance


def test_support_resistance_basic_columns():
    high = pd.Series([10.0, 12.0, 14.0, 13.0, 12.0, 13.5, 12.5])
    low = pd.Series([8.0, 9.0, 10.0, 9.5, 9.0, 9.8, 9.2])
    close = pd.Series([9.0, 11.0, 13.0, 10.5, 11.5, 13.0, 10.0])

    result = support_resistance(high, low, close, left=2, right=2)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "PIVOT_HIGH",
        "PIVOT_LOW",
        "RESISTANCE",
        "SUPPORT",
        "BREAK_RESISTANCE",
        "BREAK_SUPPORT",
    ]
    assert len(result) == len(close)


def test_support_resistance_break_flags_boolean():
    high = pd.Series([10.0, 12.0, 14.0, 13.0, 12.0, 13.5, 12.5])
    low = pd.Series([8.0, 9.0, 10.0, 9.5, 9.0, 9.8, 9.2])
    close = pd.Series([9.0, 11.0, 13.0, 10.5, 11.5, 13.0, 10.0])

    result = support_resistance(high, low, close, left=2, right=2)
    assert result["BREAK_RESISTANCE"].dtype == bool
    assert result["BREAK_SUPPORT"].dtype == bool


def test_support_resistance_input_validation():
    with pytest.raises(ValueError):
        support_resistance(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            left=0,
            right=2,
        )

    with pytest.raises(TypeError):
        support_resistance([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
