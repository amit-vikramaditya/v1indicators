import numpy as np
import pandas as pd
import pytest

from v1indicators.momentum.wavetrend import wavetrend


def test_wavetrend_basic():
    high = pd.Series([11.0, 12.0, 13.0, 14.0, 13.0, 14.0, 15.0])
    low = pd.Series([9.0, 10.0, 11.0, 12.0, 11.5, 12.5, 13.0])
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 12.2, 13.2, 14.0])

    result = wavetrend(high, low, close, channel_length=3, average_length=4, signal_length=2)

    ap = (high + low + close) / 3.0
    esa = ap.ewm(span=3, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=3, adjust=False).mean().replace(0.0, np.nan)
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=4, adjust=False).mean()
    wt2 = wt1.rolling(2).mean()
    hist = wt1 - wt2

    expected = pd.DataFrame({"WT1": wt1, "WT2": wt2, "WT_HIST": hist})
    pd.testing.assert_frame_equal(result, expected)


def test_wavetrend_input_validation():
    with pytest.raises(ValueError):
        wavetrend(
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            pd.Series([1.0, 2.0]),
            channel_length=0,
        )

    with pytest.raises(TypeError):
        wavetrend([1.0, 2.0], pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
