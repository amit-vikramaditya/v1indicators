"""Parameter-space causality sweep.

Companion to ``test_causality.py``: the main harness verifies every public
function with DEFAULT parameters. A repaint can hide behind a default (e.g.
a compensation shift sized for the default window only), so this gate re-runs
the same prefix-invariance property with each sweepable parameter pushed away
from its default — short/long windows, alternate modes, step sizes.

Functions whose signature cannot be auto-invoked are skipped, exactly as in
the main harness. A variant that raises for BOTH the full series and the
prefix is a parameter-validation rejection (fine); raising on one but not
the other, or differing prefix values, is a failure.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import v1indicators as vi

from test_causality import _equal_prefix, _synthetic_ohlcv

_PREFIXES = (400, 690)

# One axis of variation per parameter name; values chosen to stress window
# sizes, modes and grouping far from the defaults.
_PARAM_VARIANTS = {
    "length": [2, 5, 50],
    "window": [2, 5, 50],
    "period": [3, 50],
    "lookback": [3, 50],
    "left": [3, 30],
    "right": [3, 30],
    "mult": [0.5, 4.0],
    "multiplier": [0.5, 4.0],
    "threshold": [1.0],
    "mamode": ["sma", "rma"],
    "ma_type": ["wma"],
    "step": [4, 30],
    "htf_step": [4, 30],
    "drift": [2],
    "q": [0.25],
    "sensitivity": [2, 16],
    "min_strength": [1],
    "loopback": [40],
    "pivot_period": [4],
    "channel_width_pct": [1.0],
    "setup_length": [4, 25],
    "atr_length": [3],
    "trend_bars": [2],
}


def _slice(obj, k):
    if isinstance(obj, dict):
        return {key: _slice(v, k) for key, v in obj.items()}
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.iloc[:k]
    if isinstance(obj, np.ndarray):
        return obj[:k]
    return obj


def _variant_cases():
    data = _synthetic_ohlcv()
    cases = []
    for name in sorted(vi._SYMBOL_TO_PACKAGE):
        func = getattr(vi, name)
        sig = inspect.signature(func)
        for pname, values in _PARAM_VARIANTS.items():
            if pname not in sig.parameters:
                continue
            for value in values:
                cases.append(pytest.param(func, sig, pname, value, data,
                                          id=f"{name}-{pname}={value!r}"))
    return cases


@pytest.mark.parametrize("func,sig,pname,value,data", _variant_cases())
def test_prefix_invariance_under_param(func, sig, pname, value, data):
    kwargs = {}
    has_price = any(
        p in ("open", "open_", "high", "low", "close", "source", "volume")
        for p in sig.parameters
    )
    for p, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p == pname:
            kwargs[p] = value
        elif p in ("open", "open_", "high", "low", "close", "volume", "source"):
            kwargs[p] = data[p]
        elif p in ("signal", "fast", "slow") and not has_price:
            kwargs[p] = data[p]
        elif param.default is inspect.Parameter.empty:
            pytest.skip(f"required param {p!r} not synthesizable")

    try:
        out_full = func(**kwargs)
    except Exception as e:  # parameter combination rejected
        pytest.skip(f"{type(e).__name__}: {e}")

    outs_full = out_full if isinstance(out_full, tuple) else (out_full,)

    for k in _PREFIXES:
        kwargs_pre = {
            p: (v.iloc[:k] if isinstance(v, pd.Series) else v)
            for p, v in kwargs.items()
        }
        try:
            out_pre = func(**kwargs_pre)
        except Exception as e:
            raise AssertionError(
                f"{func.__name__}({pname}={value!r}) raises on prefix={k} "
                f"but not on the full series: {type(e).__name__}: {e}"
            )
        outs_pre = out_pre if isinstance(out_pre, tuple) else (out_pre,)
        for i, (a, b) in enumerate(zip(
            (_slice(o, k) for o in outs_full), outs_pre
        )):
            assert _equal_prefix(a, b), (
                f"{func.__name__}({pname}={value!r}) repaints at prefix={k} "
                f"output[{i}]: values on the first {k} bars change when "
                "future bars are appended (look-ahead bias)"
            )
