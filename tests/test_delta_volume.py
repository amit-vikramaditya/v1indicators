import pandas as pd
import pytest

from v1indicators.volume import delta_volume


def test_delta_volume_basic():
    open_ = pd.Series([10.0, 10.0, 10.0])
    close = pd.Series([11.0, 9.0, 10.0])
    volume = pd.Series([100.0, 50.0, 20.0])

    result = delta_volume(open_, close, volume)
    assert result["DELTA_VOLUME"].tolist() == [100.0, -50.0, 0.0]
    assert result["CUM_DELTA_VOLUME"].tolist() == [100.0, 50.0, 50.0]


def test_delta_volume_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(TypeError):
        delta_volume([1.0, 2.0], s, s)
