import pandas as pd
import pytest

from v1indicators.volume.vpt import vpt


def test_vpt_basic():
    close = pd.Series([10.0, 11.0, 10.0, 12.0, 12.0])
    volume = pd.Series([100.0, 150.0, 120.0, 130.0, 90.0])

    result = vpt(close, volume)
    expected = (volume * close.pct_change().fillna(0.0)).cumsum()
    expected.name = "VPT"

    pd.testing.assert_series_equal(result, expected)


def test_vpt_constant_close_zero_line():
    close = pd.Series([10.0, 10.0, 10.0, 10.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0])

    result = vpt(close, volume)
    assert (result == 0.0).all()


def test_vpt_input_validation():
    with pytest.raises(TypeError):
        vpt([1.0, 2.0], pd.Series([1.0, 2.0]))

    with pytest.raises(TypeError):
        vpt(pd.Series([1.0, 2.0]), [1.0, 2.0])
