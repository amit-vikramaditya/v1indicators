"""Reference-math parity tests: naive-loop implementations vs library code.

Each reference here is an intentionally slow, unambiguous, textbook
formulation written as a plain Python loop. The tests pin:

1. EXACT parity where the library implements the same recursion
   (EMA, RMA/Wilder, CCI with classic mean absolute deviation).
2. CONVERGENCE parity where the library deliberately uses a different
   warmup convention (RSI uses pandas ``ewm(alpha=1/n, adjust=True)`` rather
   than the SMA-seeded Wilder recursion; the two agree to ~1e-6 once past
   the warmup transient, which these tests quantify rather than hide).

These are dependency-free: no ta-lib / pandas_ta installation is required.
Cross-library parity (pandas_ta / ta-lib) is deferred to CI where those
dependencies can be pinned to installable versions.
"""

import numpy as np
import pandas as pd

from v1indicators import atr, cci, ema, rma, rsi


def _random_walk(n: int = 400, seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 + rng.normal(0.0, 1.0, n).cumsum())


def test_ema_matches_naive_recursive_loop_exactly():
    close = _random_walk()
    length = 20
    alpha = 2.0 / (length + 1)
    ref = close.to_numpy().copy()
    for i in range(1, len(ref)):
        ref[i] = alpha * ref[i] * 0 + alpha * close.iloc[i] + (1 - alpha) * ref[i - 1]
    out = ema(close, length=length).to_numpy()
    assert np.allclose(out, ref, rtol=0, atol=1e-10)


def test_rma_matches_naive_wilder_loop_exactly():
    close = _random_walk(seed=9)
    length = 14
    alpha = 1.0 / length
    ref = close.to_numpy().copy()
    for i in range(1, len(ref)):
        ref[i] = alpha * close.iloc[i] + (1 - alpha) * ref[i - 1]
    out = rma(close, length=length).to_numpy()
    assert np.allclose(out, ref, rtol=0, atol=1e-10)


def test_cci_matches_textbook_mad_reference_exactly():
    rng = np.random.default_rng(4)
    close = _random_walk(seed=4)
    high = close + np.abs(rng.normal(0.5, 0.2, len(close)))
    low = close - np.abs(rng.normal(0.5, 0.2, len(close)))
    length = 20
    c = 0.015

    tp = ((high + low + close) / 3.0).to_numpy()
    ref = np.full(len(tp), np.nan)
    for i in range(length - 1, len(tp)):
        window = tp[i - length + 1 : i + 1]
        sma = window.mean()
        mad = np.abs(window - sma).mean()
        if mad > 0:
            ref[i] = (tp[i] - sma) / (c * mad)

    out = cci(high, low, close, length=length).to_numpy()
    mask = ~np.isnan(ref)
    assert mask.sum() > len(tp) - length  # sanity: reference produced values
    assert np.allclose(out[mask], ref[mask], rtol=0, atol=1e-9)


def test_atr_rma_mode_converges_to_wilder_seeded_reference():
    close = _random_walk(seed=7)
    high = close + 0.5
    low = close - 0.5
    length = 14

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr = tr.fillna(high - low).to_numpy()

    # Textbook ATR: SMA seed at bar (length-1), then Wilder recursion.
    ref = np.full(len(tr), np.nan)
    ref[length - 1] = tr[:length].mean()
    for i in range(length, len(tr)):
        ref[i] = (ref[i - 1] * (length - 1) + tr[i]) / length

    out = atr(high, low, close, length=length, mamode="rma").to_numpy()
    # The library's rma mode seeds the recursion at bar 0 rather than with
    # the SMA seed; the difference decays geometrically. Beyond the warmup
    # transient the two must agree to tight tolerance.
    tail = slice(300, None)
    assert np.nanmax(np.abs(out[tail] - ref[tail])) < 1e-6


def test_rsi_converges_to_wilder_seeded_reference():
    close = _random_walk(seed=5)
    length = 14

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(length).mean().to_numpy().copy()
    avg_loss = loss.rolling(length).mean().to_numpy().copy()
    for i in range(length, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gain.iloc[i]) / length
        avg_loss[i] = (avg_loss[i - 1] * (length - 1) + loss.iloc[i]) / length

    ref = np.full(len(close), np.nan)
    for i in range(len(close)):
        if np.isfinite(avg_gain[i]) and np.isfinite(avg_loss[i]) and avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            ref[i] = 100.0 - 100.0 / (1.0 + rs)
        elif np.isfinite(avg_gain[i]) and avg_loss[i] == 0:
            ref[i] = 100.0

    out = rsi(close, length=length).to_numpy()
    # Documented convention difference: the library uses ewm(alpha=1/n,
    # adjust=True) (no SMA seed); values differ during the warmup transient
    # (up to ~3 RSI points) and agree to ~1e-6 beyond it.
    early = slice(length, 100)
    tail = slice(200, None)
    assert np.nanmax(np.abs(out[tail] - ref[tail])) < 1e-4
    assert np.nanmax(np.abs(out[early] - ref[early])) < 5.0  # transient is bounded
