import pandas as pd
import pytest

from v1indicators.trend import ut_bot


def test_ut_bot_basic_shape_and_columns():
    high = pd.Series([10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.1, 10.9])
    low = pd.Series([9.5, 9.8, 10.2, 10.0, 10.5, 10.9, 10.6, 10.3])
    close = pd.Series([9.8, 10.2, 10.7, 10.4, 10.9, 11.2, 10.8, 10.5])

    result = ut_bot(high, low, close, key_value=1.0, atr_period=3)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["UT_STOP", "UT_DIR", "UT_BUY", "UT_SELL"]
    assert len(result) == len(close)


def test_ut_bot_emits_direction_values():
    high = pd.Series([10, 11, 12, 13, 12, 11, 10, 9], dtype=float)
    low = pd.Series([8, 9, 10, 11, 10, 9, 8, 7], dtype=float)
    close = pd.Series([9, 10, 11, 12, 11, 10, 9, 8], dtype=float)

    result = ut_bot(high, low, close, key_value=1.0, atr_period=2)
    assert set(result["UT_DIR"].dropna().unique()).issubset({-1, 0, 1})


def test_ut_bot_input_validation():
    with pytest.raises(ValueError):
        ut_bot(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            key_value=0.0,
        )

    with pytest.raises(ValueError):
        ut_bot(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            atr_period=0,
        )

    with pytest.raises(TypeError):
        ut_bot([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
