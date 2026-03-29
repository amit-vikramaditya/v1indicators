import numpy as np
import pandas as pd
import pytest
from typing import Any, cast

from v1indicators.momentum.directional_logistic_oscillator import directional_logistic_oscillator


def _sample_ohlc(size: int = 400) -> tuple[pd.Series, pd.Series, pd.Series]:
    np.random.seed(7)
    base = 100 + np.cumsum(np.random.normal(0.0, 1.0, size))
    close = pd.Series(base)
    high = close + np.abs(np.random.normal(0.6, 0.2, size))
    low = close - np.abs(np.random.normal(0.6, 0.2, size))
    return high, low, close


def test_dlo_shape_columns_and_bounds():
    high, low, close = _sample_ohlc(450)

    result = directional_logistic_oscillator(
        high,
        low,
        close,
        di_length=14,
        mean_lookback=60,
        slope=0.18,
        probability_smoothing=3,
        oscillator_scale=2.5,
        oscillator_smoothing=7,
    )

    expected_columns = [
        "DLO_STRENGTH",
        "DLO_SMA",
        "DLO_EMA",
        "DLO_SMA_CYCLE",
        "DLO_LOWER_SMA",
        "DLO_UPPER_SMA",
        "DLO_LOWER_EMA",
        "DLO_UPPER_EMA",
        "DLO_MR_BUY",
        "DLO_MR_SELL",
        "DLO_REV_UP",
        "DLO_REV_DOWN",
    ]

    assert list(result.columns) == expected_columns
    assert len(result) == len(close)

    bounded = result["DLO_STRENGTH"].dropna()
    assert not bounded.empty
    assert (bounded <= 1.0 + 1e-12).all()
    assert (bounded >= -1.0 - 1e-12).all()

    assert result["DLO_MR_BUY"].dtype == bool
    assert result["DLO_MR_SELL"].dtype == bool
    assert result["DLO_REV_UP"].dtype == bool
    assert result["DLO_REV_DOWN"].dtype == bool


def test_dlo_validation():
    high, low, close = _sample_ohlc(120)

    with pytest.raises(ValueError):
        directional_logistic_oscillator(high, low, close, di_length=0)

    with pytest.raises(ValueError):
        directional_logistic_oscillator(high, low, close, mean_lookback=1)

    with pytest.raises(ValueError):
        directional_logistic_oscillator(high, low, close, slope=0)

    with pytest.raises(ValueError):
        directional_logistic_oscillator(high, low, close, probability_smoothing=0)

    with pytest.raises(ValueError):
        directional_logistic_oscillator(high, low, close, oscillator_scale=0)

    with pytest.raises(ValueError):
        directional_logistic_oscillator(high, low, close, oscillator_smoothing=0)

    with pytest.raises(TypeError):
        directional_logistic_oscillator(cast(Any, [1.0, 2.0]), low, close)
