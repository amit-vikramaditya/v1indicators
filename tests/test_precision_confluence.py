import numpy as np
import pandas as pd
import pytest
from typing import Any, cast

from v1indicators.trend.precision_confluence import precision_confluence


def _ohlcv(n: int = 260) -> pd.DataFrame:
    np.random.seed(31)
    close = pd.Series(120 + np.cumsum(np.random.normal(0, 0.7, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_.to_numpy(), close.to_numpy()) + np.abs(np.random.normal(0.5, 0.15, n)))
    low = pd.Series(np.minimum(open_.to_numpy(), close.to_numpy()) - np.abs(np.random.normal(0.5, 0.15, n)))
    volume = pd.Series(np.random.randint(120, 1200, n), dtype=float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_precision_confluence_columns_and_shape():
    df = _ohlcv()
    out = precision_confluence(df["open"], df["high"], df["low"], df["close"], df["volume"], preset="default")

    needed = {
        "PC_BULL_SCORE",
        "PC_BEAR_SCORE",
        "PC_BUY",
        "PC_SELL",
        "PC_DIR",
        "PC_ENTRY",
        "PC_SL",
        "PC_TP1",
        "PC_TP3",
        "PC_PROFILE",
    }
    assert needed.issubset(set(out.columns))
    assert len(out) == len(df)
    assert out["PC_BUY"].dtype == bool
    assert out["PC_SELL"].dtype == bool
    assert (out["PC_PROFILE"] == "default").any()


def test_precision_confluence_validation():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        precision_confluence(s, s, s, s, s, swing_lookback=0)
    with pytest.raises(ValueError):
        precision_confluence(s, s, s, s, s, tp1_mult=0)
    with pytest.raises(ValueError):
        precision_confluence(s, s, s, s, s, preset="unknown")
    with pytest.raises(TypeError):
        precision_confluence(cast(Any, [1.0, 2.0]), s, s, s, s)
