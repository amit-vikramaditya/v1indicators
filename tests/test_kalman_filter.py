import numpy as np
import pandas as pd
import pytest

from v1indicators.overlap.kalman_filter import kalman_filter


def _expected_kalman(source, high, low, close, velocity_alpha, range_alpha, memory_alpha):
    src = source.to_numpy(dtype=np.float64)
    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    c = close.to_numpy(dtype=np.float64)

    out = np.full(src.shape, np.nan, dtype=np.float64)
    out[0] = src[0]

    value1 = 0.0
    value2 = 0.0

    for i in range(1, src.size):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        value1 = velocity_alpha * (src[i] - src[i - 1]) + memory_alpha * value1
        value2 = range_alpha * tr + memory_alpha * value2
        lam = abs(value1 / value2) if value2 != 0.0 else 0.0
        alpha = (-lam * lam + np.sqrt(lam**4 + 16.0 * lam * lam)) / 8.0
        out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1]

    return pd.Series(out, index=source.index, name="KALMAN_FILTER")


def test_kalman_filter_basic():
    source = pd.Series([10.0, 10.4, 10.1, 10.6, 10.9, 10.7, 11.0])
    high = pd.Series([10.3, 10.6, 10.5, 10.8, 11.1, 11.0, 11.2])
    low = pd.Series([9.8, 10.1, 9.9, 10.2, 10.6, 10.4, 10.7])
    close = pd.Series([10.1, 10.3, 10.0, 10.7, 10.8, 10.6, 11.1])

    result = kalman_filter(source, high, low, close)
    expected = _expected_kalman(source, high, low, close, 0.2, 0.1, 0.8)

    pd.testing.assert_series_equal(result, expected)


def test_kalman_filter_input_validation():
    src = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError):
        kalman_filter(src, src, src, src, velocity_alpha=0.0)

    with pytest.raises(ValueError):
        kalman_filter(src, src, src, src, range_alpha=-1.0)

    with pytest.raises(ValueError):
        kalman_filter(src, src, src, src, memory_alpha=1.1)

    with pytest.raises(TypeError):
        kalman_filter([1.0, 2.0], src, src, src)
