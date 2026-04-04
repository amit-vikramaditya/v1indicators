import numpy as np
import pandas as pd
import pytest

from v1indicators.momentum import squeeze_momentum


def _linreg_last(window: np.ndarray) -> float:
    x = np.arange(window.size, dtype=np.float64)
    slope, intercept = np.polyfit(x, window, 1)
    return intercept + slope * (window.size - 1)


def test_squeeze_momentum_basic():
    high = pd.Series([11.0, 12.0, 13.0, 14.0, 13.5, 14.5, 15.0, 15.5, 16.0, 15.8])
    low = pd.Series([9.0, 10.0, 10.5, 11.0, 11.2, 12.0, 12.4, 12.8, 13.2, 13.0])
    close = pd.Series([10.0, 11.0, 12.0, 12.5, 12.8, 13.6, 14.0, 14.8, 15.2, 15.0])

    result = squeeze_momentum(
        high,
        low,
        close,
        bb_length=4,
        bb_mult=2.0,
        kc_length=4,
        kc_mult=1.5,
        use_true_range=True,
    )

    bb_mid = close.rolling(4).mean()
    bb_std = close.rolling(4).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    kc_mid = close.rolling(4).mean()
    kc_range_ma = tr.rolling(4).mean()
    kc_upper = kc_mid + 1.5 * kc_range_ma
    kc_lower = kc_mid - 1.5 * kc_range_ma

    expected_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    expected_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
    expected_no = ~(expected_on | expected_off)

    highest_h = high.rolling(4).max()
    lowest_l = low.rolling(4).min()
    center = ((highest_h + lowest_l) / 2.0 + close.rolling(4).mean()) / 2.0
    deviation = close - center
    expected_mom = deviation.rolling(4).apply(_linreg_last, raw=True)

    np.testing.assert_allclose(result["SQZ_MOM"], expected_mom, equal_nan=True)
    pd.testing.assert_series_equal(result["SQZ_ON"], expected_on, check_names=False)
    pd.testing.assert_series_equal(result["SQZ_OFF"], expected_off, check_names=False)
    pd.testing.assert_series_equal(result["SQZ_NO"], expected_no, check_names=False)


def test_squeeze_momentum_use_range_mode():
    high = pd.Series([10.0, 10.5, 11.0, 11.2, 11.6, 12.0])
    low = pd.Series([9.4, 9.8, 10.0, 10.2, 10.7, 11.1])
    close = pd.Series([9.8, 10.2, 10.4, 10.8, 11.2, 11.7])

    result = squeeze_momentum(
        high,
        low,
        close,
        bb_length=3,
        kc_length=3,
        use_true_range=False,
    )

    assert list(result.columns) == ["SQZ_MOM", "SQZ_ON", "SQZ_OFF", "SQZ_NO"]
    assert len(result) == len(close)


def test_squeeze_momentum_input_validation():
    with pytest.raises(ValueError):
        squeeze_momentum(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            bb_length=1,
            kc_length=2,
        )

    with pytest.raises(ValueError):
        squeeze_momentum(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            bb_length=2,
            kc_length=2,
            bb_mult=0.0,
        )

    with pytest.raises(TypeError):
        squeeze_momentum([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
