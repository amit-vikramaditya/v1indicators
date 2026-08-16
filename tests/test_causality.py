"""Prefix-invariance causality harness for every public v1indicators function.

Property under test
-------------------
An indicator is *causal* (free of look-ahead bias / repainting) when its
values on bars [0:K] are IDENTICAL whether it is computed on the full series
(N bars) or on the truncated prefix (first K bars). If appending future bars
changes any past value, the indicator repaints and any backtest built on it
is inflated.

This is a whole-library quality gate: every public function reachable from
the package root is auto-discovered, auto-invoked on synthetic OHLCV data
by signature introspection, and checked for prefix-invariance.

Known look-ahead by design (documented, intentionally not causal):
    - ``ichimoku``: the Chikou span is today's close plotted kijun bars back
      by definition.

Every other function MUST pass. If a fix regresses causality, this file
fails; if a new indicator repaints, this file fails. Do not add entries to
KNOWN_LOOKAHEAD without an explicit mathematical justification in the reason
string and a decision recorded in the changelog.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import v1indicators as vi

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

_N_BARS = 800
# Multiple cut points: a repainting indicator only reveals itself when a
# repainted feature (e.g. an unconfirmed pivot) falls within `right` bars of
# the cut. A single prefix can miss it by luck; five cuts spanning 40%-87%
# of the series make that practically impossible.
_PREFIXES = (320, 400, 480, 560, 690)
_SEED = 20260816


def _synthetic_ohlcv() -> dict[str, pd.Series]:
    """Deterministic OHLCV with trends, gaps, flat stretches and NaN-free wicks."""
    rng = np.random.default_rng(_SEED)
    index = pd.date_range(
        start=datetime(2026, 3, 2, 9, 15),  # a Monday
        periods=_N_BARS,
        freq="15min",
    )

    drift = np.concatenate(
        [
            np.linspace(0.0, 30.0, _N_BARS // 4),
            np.linspace(30.0, -20.0, _N_BARS // 4),
            np.full(_N_BARS // 4, -5.0),
            np.linspace(-5.0, 25.0, _N_BARS - 3 * (_N_BARS // 4)),
        ]
    )
    noise = rng.normal(0.0, 1.1, _N_BARS).cumsum() * 0.35
    close = pd.Series(5000.0 + drift + noise, index=index, name="close")

    open_ = close.shift(1).fillna(close.iloc[0])
    wick_up = np.abs(rng.normal(0.9, 0.4, _N_BARS))
    wick_dn = np.abs(rng.normal(0.9, 0.4, _N_BARS))
    high = pd.Series(
        np.maximum(open_.to_numpy(), close.to_numpy()) + wick_up,
        index=index,
        name="high",
    )
    low = pd.Series(
        np.minimum(open_.to_numpy(), close.to_numpy()) - wick_dn,
        index=index,
        name="low",
    )
    volume = pd.Series(
        rng.integers(5_000, 250_000, _N_BARS).astype(np.float64),
        index=index,
        name="volume",
    )

    fast = close.ewm(span=10, adjust=False).mean()
    slow = close.ewm(span=30, adjust=False).mean()

    return {"open": open_, "open_": open_, "high": high, "low": low,
            "close": close, "source": close, "volume": volume,
            "signal": close, "fast": fast, "slow": slow}


# ---------------------------------------------------------------------------
# Auto-discovery + signature-driven invocation
# ---------------------------------------------------------------------------

_SERIES_PARAM_ALIASES = {
    "open": "open",
    "open_": "open_",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "source": "source",  # kalman_filter: any price series works
    "signal": "signal",  # decay(): generic trigger series
    "fast": "fast",      # long_run/short_run: fast MA series
    "slow": "slow",      # long_run/short_run: slow MA series
}


def _public_symbols() -> list[str]:
    return sorted(vi._SYMBOL_TO_PACKAGE.keys())


def _invoke(func, data: dict[str, pd.Series]):
    """Call ``func`` with series mapped from its parameter names, defaults elsewhere.

    Returns (result, None) on success or (None, reason) when a required
    parameter cannot be satisfied from OHLCV series (auto-skip case).

    ``fast``/``slow``/``signal`` are ambiguous: MA-pair/trigger functions
    (long_run/short_run/decay) take them as SERIES, while oscillator
    functions (ao, macd, kdj, ...) take them as integer lengths. Disambiguated
    by context: if the signature already receives any price/volume series,
    they are scalars.
    """
    sig = inspect.signature(func)
    has_price_series = any(
        p in ("open", "open_", "high", "low", "close", "source", "volume")
        for p in sig.parameters
    )
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in _SERIES_PARAM_ALIASES:
            if name in ("fast", "slow", "signal") and has_price_series:
                continue  # integer length in this signature; keep its default
            kwargs[name] = data[_SERIES_PARAM_ALIASES[name]]
        elif param.default is inspect.Parameter.empty:
            # A required parameter we cannot supply (e.g. a second index,
            # a signal series, a DataFrame-based API).
            return None, f"required param {name!r} not synthesizable"
    return func(**kwargs), None


# ---------------------------------------------------------------------------
# Prefix-invariance comparison
# ---------------------------------------------------------------------------

def _equal_prefix(a, b) -> bool:
    """NaN-aware exact equality of two aligned prefix outputs."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_equal_prefix(a[k], b[k]) for k in a)
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        if list(a.columns) != list(b.columns):
            return False
        return all(_equal_prefix(a[c], b[c]) for c in a.columns)
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        if len(a) != len(b):
            return False
        av, bv = a.to_numpy(), b.to_numpy()
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        av, bv = a, b
        if av.shape != bv.shape:
            return False
    else:
        return bool(a == b)

    if av.dtype == object or bv.dtype == object:
        # Numeric object columns (pd.NA leakage) normalize to float; genuine
        # string columns compare elementwise.
        try:
            av = pd.array(av, dtype="Float64").to_numpy(dtype=np.float64, na_value=np.nan)
            bv = pd.array(bv, dtype="Float64").to_numpy(dtype=np.float64, na_value=np.nan)
        except (ValueError, TypeError):
            return bool(np.array_equal(av.astype(str), bv.astype(str)))
    if av.dtype == bool or bv.dtype == bool:
        return bool(np.array_equal(av.astype(bool), bv.astype(bool)))
    try:
        return bool(np.allclose(av.astype(np.float64), bv.astype(np.float64),
                                rtol=0.0, atol=0.0, equal_nan=True))
    except (TypeError, ValueError):
        return bool(np.array_equal(av, bv))


def _to_float(obj):
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, np.ndarray):
        return pd.Series(obj)
    return obj


def _check_prefix_invariance(func) -> None:
    full = _synthetic_ohlcv()

    out_full, skip_reason = _invoke(func, full)
    if skip_reason:
        pytest.skip(skip_reason)

    def _slice(obj, k):
        if isinstance(obj, dict):
            return {key: _slice(v, k) for key, v in obj.items()}
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[:k]
        if isinstance(obj, np.ndarray):
            return obj[:k]
        return obj

    outs_full = out_full if isinstance(out_full, tuple) else (out_full,)

    for k in _PREFIXES:
        prefix = {name: s.iloc[:k] for name, s in full.items()}
        out_prefix, skip_reason2 = _invoke(func, prefix)
        if skip_reason2:  # pragma: no cover - defensive
            pytest.skip(skip_reason2)
        outs_prefix = out_prefix if isinstance(out_prefix, tuple) else (out_prefix,)

        for i, (a, b) in enumerate(zip((_slice(o, k) for o in outs_full), outs_prefix)):
            assert _equal_prefix(a, b), (
                f"output[{i}] repaints at prefix={k}: values on the first {k} "
                "bars change when future bars are appended (look-ahead bias)"
            )


# ---------------------------------------------------------------------------
# Known look-ahead by design — the ONLY permitted failures
# ---------------------------------------------------------------------------

KNOWN_LOOKAHEAD = {
    "ichimoku": "Chikou span is today's close plotted kijun bars back by "
                "definition (textbook Ichimoku construction).",
}

# Whole-window SNAPSHOT functions: their output is a summary of the entire
# input by definition (not a per-bar indicator), so prefix-invariance does
# not apply. Currently empty — the one snapshot function (vp) was removed
# in 1.0.0. The mechanism is kept for any future snapshot-style API.
SNAPSHOT_FUNCTIONS = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    pytest.param(
        name,
        marks=pytest.mark.skip(reason=SNAPSHOT_FUNCTIONS[name])
    ) if name in SNAPSHOT_FUNCTIONS else
    pytest.param(
        name,
        marks=pytest.mark.xfail(reason=KNOWN_LOOKAHEAD[name], strict=True)
    ) if name in KNOWN_LOOKAHEAD else name
    for name in _public_symbols()
])
def test_prefix_invariance(name):
    func = getattr(vi, name)
    assert callable(func)
    _check_prefix_invariance(func)


def test_known_lookahead_registry_covers_only_real_entries():
    """Guard against stale registry entries: every listed name must exist and
    must indeed fail causality (strict xfail). If someone fixes a KNOWN
    indicator without removing it here, the strict xfail above turns into
    an XPASS and fails loudly — this test just documents the contract."""
    for name in KNOWN_LOOKAHEAD:
        assert name in vi._SYMBOL_TO_PACKAGE, f"{name} no longer public"
