import numpy as np
import pandas as pd
import pytest

from v1indicators.performance import drawdown
from v1indicators.performance import log_return
from v1indicators.performance import percent_return


def test_log_and_percent_return_basic():
    close = pd.Series([10.0, 11.0, 12.1, 11.0])

    lr = log_return(close, cumulative=False, length=1)
    pr = percent_return(close, cumulative=False, length=1)

    exp_lr = np.log(close / close.shift(1)).rename("LOGRET_1")
    exp_pr = close.pct_change(1).rename("PCTRET_1")

    pd.testing.assert_series_equal(lr, exp_lr)
    pd.testing.assert_series_equal(pr, exp_pr)


def test_drawdown_basic():
    close = pd.Series([10.0, 12.0, 11.0, 13.0, 9.0])
    result = drawdown(close)

    peak = close.cummax()
    exp_abs = close - peak
    exp_pct = close / peak - 1.0
    exp_log = np.log(close / peak)

    pd.testing.assert_series_equal(result["DRAWDOWN"], exp_abs, check_names=False)
    pd.testing.assert_series_equal(result["DRAWDOWN_PCT"], exp_pct, check_names=False)
    pd.testing.assert_series_equal(result["DRAWDOWN_LOG"], exp_log, check_names=False)


def test_performance_input_validation():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        log_return(s, length=0)
    with pytest.raises(ValueError):
        percent_return(s, length=0)
    with pytest.raises(TypeError):
        log_return([1.0, 2.0])
