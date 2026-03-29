import numpy as np
import pandas as pd
import pytest
from typing import Any, cast

from v1indicators.trend.dual_score_signals import dual_score_signals


def _ohlcv(n: int = 240) -> pd.DataFrame:
    np.random.seed(11)
    base = 100 + np.cumsum(np.random.normal(0, 0.6, n))
    close = pd.Series(base)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_.to_numpy(), close.to_numpy()) + np.abs(np.random.normal(0.3, 0.1, n)))
    low = pd.Series(np.minimum(open_.to_numpy(), close.to_numpy()) - np.abs(np.random.normal(0.3, 0.1, n)))
    volume = pd.Series(np.random.randint(100, 1000, n), dtype=float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_dual_score_signals_columns_and_shape():
    df = _ohlcv()
    out = dual_score_signals(df["open"], df["high"], df["low"], df["close"], df["volume"])

    required = {
        "DSS_BULL_SCORE",
        "DSS_BEAR_SCORE",
        "DSS_BUY",
        "DSS_SELL",
        "DSS_STATE",
        "DSS_ENTRY",
        "DSS_SL",
        "DSS_TP1",
        "DSS_TP5",
        "DSS_RETEST",
    }
    assert required.issubset(set(out.columns))
    assert len(out) == len(df)
    assert out["DSS_BUY"].dtype == bool
    assert out["DSS_SELL"].dtype == bool
    assert not ((out["DSS_BUY"] & out["DSS_SELL"]).any())


def test_dual_score_signals_validation():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        dual_score_signals(s, s, s, s, s, ema_fast=0)
    with pytest.raises(ValueError):
        dual_score_signals(s, s, s, s, s, mtf_step=0)
    with pytest.raises(ValueError):
        dual_score_signals(s, s, s, s, s, atr_multiplier=0)
    with pytest.raises(TypeError):
        dual_score_signals(cast(Any, [1.0, 2.0]), s, s, s, s)
