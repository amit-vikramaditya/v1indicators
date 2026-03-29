import numpy as np
import pandas as pd
import pytest
from typing import Any, cast

from v1indicators.volume.swing_leg_profile import swing_leg_profile


def _ohlcv(n: int = 320) -> pd.DataFrame:
    np.random.seed(21)
    x = np.linspace(0, 8 * np.pi, n)
    close = pd.Series(100.0 + np.sin(x) * 4.0 + np.random.normal(0, 0.2, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_.to_numpy(), close.to_numpy()) + 0.4)
    low = pd.Series(np.minimum(open_.to_numpy(), close.to_numpy()) - 0.4)
    volume = pd.Series(np.random.randint(80, 800, n), dtype=float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_swing_leg_profile_columns_and_values():
    df = _ohlcv()
    out = swing_leg_profile(
        df["open"],
        df["high"],
        df["low"],
        df["close"],
        df["volume"],
        swing_length=20,
        atr_length=30,
    )

    expected = {
        "SLP_DIR",
        "SLP_POC",
        "SLP_TOTAL_VOL",
        "SLP_DELTA_PCT",
        "SLP_BIN_COUNT",
    }
    assert expected.issubset(set(out.columns))
    assert len(out) == len(df)
    assert set(out["SLP_DIR"].dropna().unique()).issubset({-1, 1})
    assert out["SLP_TOTAL_VOL"].notna().sum() > 0


def test_swing_leg_profile_validation():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        swing_leg_profile(s, s, s, s, s, swing_length=1)
    with pytest.raises(ValueError):
        swing_leg_profile(s, s, s, s, s, bin_atr_mult=0)
    with pytest.raises(TypeError):
        swing_leg_profile(cast(Any, [1.0, 2.0]), s, s, s, s)
