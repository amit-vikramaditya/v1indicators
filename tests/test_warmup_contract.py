"""Warmup and input-contract tests.

The library's warmup convention (1.0.0): every indicator is NaN until it has
enough history to produce a meaningful value. Exponential-family outputs do
NOT emit bar-0 transient estimates; rolling outputs are NaN until their
window fills. This pins the convention for the core primitives so it cannot
silently regress.
"""

import numpy as np
import pandas as pd
import pytest

from v1indicators import atr, dema, ema, macd, rma, smma, stoch, t3, tema, williams_r, zlema
from v1indicators._utils import check_series


def _walk(n=120, seed=11):
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 + rng.normal(0, 1, n).cumsum())


@pytest.mark.parametrize("length", [5, 10, 20])
def test_ema_nan_until_length_bars(length):
    out = ema(_walk(), length=length)
    assert out.iloc[: length - 1].isna().all()
    assert not np.isnan(out.iloc[length - 1])


@pytest.mark.parametrize("func", [rma, smma])
def test_wilder_family_warmup(func):
    out = func(_walk(), length=10)
    assert out.iloc[:9].isna().all()
    assert not np.isnan(out.iloc[9])


def test_zlema_warmup_includes_lag_shift():
    # lag = (10-1)//2 = 4 NaN inputs, then 10 ewm observations: first valid
    # at bar 4 + 9 = 13.
    out = zlema(_walk(), length=10)
    assert out.iloc[:13].isna().all()
    assert not np.isnan(out.iloc[13])


def test_nested_ema_warmup_composes():
    close = _walk()
    d = dema(close, length=5)
    t = tema(close, length=5)
    assert d.iloc[:8].isna().all() and not np.isnan(d.iloc[8])   # 2*(5-1)
    assert t.iloc[:12].isna().all() and not np.isnan(t.iloc[12])  # 3*(5-1)
    x = t3(close, length=5)
    assert x.iloc[:24].isna().all() and not np.isnan(x.iloc[24])  # 6*(5-1)


def test_macd_warmup_inherits_from_ema():
    out = macd(_walk(), fast=12, slow=26, signal=9)
    # MACD line: NaN until slow-1 (bar 25). Signal: + signal-1 more (bar 33).
    assert out["MACD"].iloc[:25].isna().all()
    assert out["MACD_SIGNAL"].iloc[:33].isna().all()
    assert not np.isnan(out["MACD_SIGNAL"].iloc[33])


def test_rolling_indicators_keep_window_warmup():
    close = _walk()
    high, low = close + 0.5, close - 0.5
    k = stoch(high, low, close, length=14, smooth=3)
    assert k["STOCH_K"].iloc[:13].isna().all()
    assert k["STOCH_D"].iloc[:15].isna().all()  # 13 + (3-1)
    w = williams_r(high, low, close, length=14)
    assert w.iloc[:13].isna().all()
    a = atr(high, low, close, length=14)
    assert a.iloc[:13].isna().all()


def test_check_series_rejects_unsorted_index():
    idx = pd.DatetimeIndex(["2026-01-03", "2026-01-01", "2026-01-02"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    with pytest.raises(ValueError, match="sorted in ascending order"):
        check_series(s, "close")
    # ascending (ties allowed) is fine
    ok = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(
        ["2026-01-01", "2026-01-02", "2026-01-02"]))
    assert check_series(ok, "close") is not None


def test_indicator_rejects_unsorted_index():
    idx = pd.DatetimeIndex(["2026-01-03", "2026-01-01", "2026-01-02"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    with pytest.raises(ValueError, match="sorted"):
        ema(s, length=2)
