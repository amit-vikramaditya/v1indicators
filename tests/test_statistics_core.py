import numpy as np
import pandas as pd
import pytest

from v1indicators.statistics import stdev
from v1indicators.statistics import variance
from v1indicators.statistics import zscore


def test_stdev_variance_zscore_basic():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])

    st = stdev(close, length=3)
    vr = variance(close, length=3)
    zs = zscore(close, length=3)

    exp_st = close.rolling(3).std(ddof=0)
    exp_vr = close.rolling(3).var(ddof=0)
    exp_zs = (close - close.rolling(3).mean()) / exp_st.replace(0.0, np.nan)

    pd.testing.assert_series_equal(st, exp_st.rename("STDEV_3"))
    pd.testing.assert_series_equal(vr, exp_vr.rename("VAR_3"))
    pd.testing.assert_series_equal(zs, exp_zs.rename("ZSCORE_3"))


def test_statistics_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        stdev(s, length=0)
    with pytest.raises(ValueError):
        variance(s, length=0)
    with pytest.raises(ValueError):
        zscore(s, length=0)
    with pytest.raises(TypeError):
        stdev([1.0, 2.0])
