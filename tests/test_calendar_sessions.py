"""Regression tests for calendar/session correctness on real-market data shapes.

The causality harness synthesizes a CONTINUOUS 24/7 index, so it cannot see
calendar-gap bugs. These tests use weekday-only indices (weekends and
holidays absent), which is how real session data looks.
"""

import numpy as np
import pandas as pd
import pytest

from v1indicators.trend import day_week_month_levels, session_killzones, session_range


def _weekday_ohlcv(start: str, end: str, freq: str = "15min", seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq=freq)
    idx = idx[idx.dayofweek < 5]
    n = len(idx)
    close = pd.Series(100.0 + rng.normal(0, 1, n).cumsum() * 0.1, index=idx)
    high = close + 0.5
    low = close - 0.5
    open_ = close.shift(1).fillna(close.iloc[0])
    return open_, high, low, close


# ---------------------------------------------------------------------------
# day_week_month_levels
# ---------------------------------------------------------------------------

def test_prior_day_levels_survive_weekends():
    """Monday bars must see FRIDAY's levels, not NaN (regression: weekend
    resample bins poisoned Monday with NaN after shift+ffill)."""
    open_, high, low, close = _weekday_ohlcv("2026-02-23", "2026-03-06")  # Mon..Fri x2

    out = day_week_month_levels(open_, high, low)

    monday = pd.Timestamp("2026-03-02").date()
    fri = pd.Timestamp("2026-02-27")
    mon_bars = out[out.index.date == monday]
    assert len(mon_bars) > 0
    assert mon_bars["PD_HIGH"].notna().all(), "Monday lost prior-day levels"
    fri_high = high.loc[fri.date().isoformat()].max()
    assert np.allclose(mon_bars["PD_HIGH"].dropna().unique(), fri_high)
    # Every trading day except the first has levels (the first day
    # legitimately has none: no prior period exists).
    from_second_day = out.loc[out.index >= "2026-02-24"]
    assert from_second_day["PD_HIGH"].notna().all()


def test_prior_day_levels_skip_holidays():
    """A missing trading day (holiday) must be skipped, not propagated as NaN."""
    open_, high, low, close = _weekday_ohlcv("2026-03-02", "2026-03-13")
    # Remove an entire Wednesday (holiday).
    holiday = (high.index.dayofweek == 2) & (high.index.date == pd.Timestamp("2026-03-11").date())
    keep = ~holiday
    open_, high, low = open_[keep], high[keep], low[keep]

    out = day_week_month_levels(open_, high, low)

    thu = out.index.date == pd.Timestamp("2026-03-12").date()
    wed = pd.Timestamp("2026-03-11")
    tue_high = high.loc[(high.index.date == pd.Timestamp("2026-03-10").date())].max()
    thu_vals = out.loc[thu, "PD_HIGH"].dropna().unique()
    assert len(thu_vals) == 1
    assert np.isclose(thu_vals[0], tue_high), "day after a holiday must see the prior TRADING day"


def test_prior_month_levels_are_last_completed_month():
    """March bars must see FEBRUARY's levels (regression: month-END labels
    made every bar see the month-before-last)."""
    open_, high, low, close = _weekday_ohlcv("2026-01-05", "2026-03-13")

    out = day_week_month_levels(open_, high, low)

    jan_high = high.loc["2026-01"].max()
    feb_high = high.loc["2026-02"].max()
    assert not np.isclose(jan_high, feb_high)
    mar_bars = out.loc[(out.index >= "2026-03-01") & (out.index < "2026-03-02")]
    assert mar_bars["PM_HIGH"].notna().all()
    assert np.allclose(mar_bars["PM_HIGH"].unique(), feb_high), (
        "March must see February (last completed month), not January"
    )


def test_prior_levels_never_use_in_progress_period():
    open_, high, low, close = _weekday_ohlcv("2026-02-02", "2026-02-27")
    out = day_week_month_levels(open_, high, low)
    # Friday afternoon bars must not see Friday's own high as PD_HIGH.
    fri_late = out.loc[(out.index.date == pd.Timestamp("2026-02-27").date())]
    fri_high = high.loc["2026-02-27"].max()
    thu_high = high.loc[(high.index.date == pd.Timestamp("2026-02-26").date())].max()
    assert np.allclose(fri_late["PD_HIGH"].dropna().unique(), thu_high)
    assert not np.isclose(fri_late["PD_HIGH"].dropna().iloc[0], fri_high)


# ---------------------------------------------------------------------------
# session_range: overnight wrap
# ---------------------------------------------------------------------------

def test_session_range_overnight_wraps_and_accumulates():
    idx = pd.date_range("2026-03-02 20:00", periods=48, freq="1h")  # 2 days
    high = pd.Series(np.linspace(10, 12, len(idx)), index=idx)
    low = high - 0.4
    close = high - 0.2

    out = session_range(high, low, close, start="22:00", end="04:00")

    active = out["SESSION_ACTIVE"]
    # 22:00-23:59 and 00:00-03:59 are active on each day
    assert active.loc["2026-03-02 22:00"]
    assert active.loc["2026-03-03 01:00"]
    assert not active.loc["2026-03-03 12:00"]

    # Running high accumulates ACROSS midnight: the 00:xx bar's session high
    # includes the 22:00-23:59 extremes of the same (previous-day) session.
    first_night_max = high.loc["2026-03-02 22:00":"2026-03-02 23:59"].max()
    after_midnight = out.loc["2026-03-03 01:00"]
    assert after_midnight["SESSION_HIGH"] >= first_night_max


def test_session_range_same_day_unchanged():
    idx = pd.date_range("2026-03-02 08:00", periods=16, freq="30min")
    high = pd.Series(np.linspace(10, 11, len(idx)), index=idx)
    low = high - 0.5
    close = high - 0.25
    out = session_range(high, low, close, start="08:30", end="12:00")
    assert out["SESSION_ACTIVE"].iloc[1]
    assert not out["SESSION_ACTIVE"].iloc[0]
    assert out["SESSION_HIGH"].iloc[1] == high.iloc[1]


# ---------------------------------------------------------------------------
# session_killzones: cross-midnight grouping
# ---------------------------------------------------------------------------

def test_session_killzones_wrapped_zone_spans_midnight():
    idx = pd.date_range("2026-03-02 21:00", periods=48, freq="1h")
    high = pd.Series(np.linspace(20, 24, len(idx)), index=idx)
    low = high - 0.3
    close = high - 0.1

    out = session_killzones(high, low, close, asia=("22:00", "02:00"))

    pre = out.loc["2026-03-02 23:00", "ASIA_HIGH"]
    post = out.loc["2026-03-03 01:00", "ASIA_HIGH"]
    pre_max = high.loc["2026-03-02 22:00":"2026-03-02 23:59"].max()
    assert post >= pre_max, "post-midnight bars must share the wrapped session's extremes"
    assert np.isclose(pre, pre_max)


def test_session_killzones_default_zones_unchanged():
    idx = pd.date_range("2026-03-02 01:00", periods=24, freq="1h")
    high = pd.Series(np.linspace(30, 31, len(idx)), index=idx)
    low = high - 0.2
    close = high - 0.1
    out = session_killzones(high, low, close)
    london_active = out["LONDON_ACTIVE"]
    assert london_active.loc["2026-03-02 03:00"]
    assert not london_active.loc["2026-03-02 06:00"]
    assert not out["ASIA_ACTIVE"].loc["2026-03-02 01:00"]
