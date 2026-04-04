import pandas as pd
import pytest

from v1indicators.momentum import ppo


def test_ppo_basic():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    result = ppo(close, fast=2, slow=4)

    ema_fast = close.ewm(span=2, adjust=False).mean()
    ema_slow = close.ewm(span=4, adjust=False).mean()
    expected = 100.0 * (ema_fast - ema_slow) / ema_slow
    expected.name = "PPO_2_4"

    pd.testing.assert_series_equal(result, expected)


def test_ppo_zero_denominator_nan():
    close = pd.Series([0.0, 0.0, 0.0, 0.0])
    result = ppo(close, fast=1, slow=2)
    assert result.isna().all()


def test_ppo_input_validation():
    with pytest.raises(ValueError):
        ppo(pd.Series([1.0, 2.0]), fast=0, slow=2)

    with pytest.raises(ValueError):
        ppo(pd.Series([1.0, 2.0]), fast=3, slow=2)

    with pytest.raises(TypeError):
        ppo([1.0, 2.0], fast=1, slow=2)
