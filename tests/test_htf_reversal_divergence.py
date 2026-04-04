import numpy as np
import pandas as pd
import pytest

from v1indicators.trend import htf_reversal_divergence


def _ohlc(n: int = 180) -> pd.DataFrame:
    np.random.seed(41)
    close = pd.Series(90 + np.cumsum(np.random.normal(0, 0.5, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_.to_numpy(), close.to_numpy()) + np.abs(np.random.normal(0.3, 0.1, n)))
    low = pd.Series(np.minimum(open_.to_numpy(), close.to_numpy()) - np.abs(np.random.normal(0.3, 0.1, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_htf_reversal_divergence_outputs():
    df = _ohlc()
    out = htf_reversal_divergence(df["open"], df["high"], df["low"], df["close"], htf_step=10)

    required = {
        "HRD_GROUP_END",
        "HRD_BULL_PATTERN",
        "HRD_BEAR_PATTERN",
        "HRD_RSI",
        "HRD_BULL_DIV",
        "HRD_BEAR_DIV",
    }
    assert required.issubset(set(out.columns))
    assert len(out) == len(df)
    assert out["HRD_GROUP_END"].sum() >= len(df) // 10
    assert out["HRD_BULL_DIV"].dtype == bool
    assert out["HRD_BEAR_DIV"].dtype == bool


def test_htf_reversal_divergence_validation():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        htf_reversal_divergence(s, s, s, s, htf_step=0)
    with pytest.raises(ValueError):
        htf_reversal_divergence(s, s, s, s, pivot_left=0)
    with pytest.raises(TypeError):
        htf_reversal_divergence([1.0, 2.0], s, s, s)
