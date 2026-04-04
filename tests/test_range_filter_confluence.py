import numpy as np
import pandas as pd
import pytest

from v1indicators.trend import range_filter_confluence


def _ohlc(n: int = 260) -> pd.DataFrame:
    np.random.seed(51)
    close = pd.Series(110 + np.cumsum(np.random.normal(0, 0.8, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_.to_numpy(), close.to_numpy()) + np.abs(np.random.normal(0.35, 0.12, n)))
    low = pd.Series(np.minimum(open_.to_numpy(), close.to_numpy()) - np.abs(np.random.normal(0.35, 0.12, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_range_filter_confluence_shape_and_flags():
    df = _ohlc()
    out = range_filter_confluence(df["high"], df["low"], df["close"])

    required = {
        "RFC_FILTER",
        "RFC_TREND",
        "RFC_LONG",
        "RFC_SHORT",
        "RFC_SCORE_STRENGTH",
        "RFC_CHOP_GATE",
    }
    assert required.issubset(set(out.columns))
    assert len(out) == len(df)
    assert out["RFC_LONG"].dtype == bool
    assert out["RFC_SHORT"].dtype == bool
    assert not (out["RFC_LONG"] & out["RFC_SHORT"]).any()


def test_range_filter_confluence_validation():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        range_filter_confluence(s, s, s, sensitivity=0)
    with pytest.raises(ValueError):
        range_filter_confluence(s, s, s, atr_multiplier=0)
    with pytest.raises(ValueError):
        range_filter_confluence(s, s, s, cooldown_bars=0)
    with pytest.raises(TypeError):
        range_filter_confluence([1.0, 2.0], s, s)
