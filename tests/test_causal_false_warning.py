"""Every indicator that exposes a `causal` parameter must warn when it is
turned off, so the retrospective mode can never be used silently in a
backtest. Auto-discovered from the public API: a new indicator with a
causal parameter that forgets the guard fails here.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import v1indicators as vi


def _data():
    rng = np.random.default_rng(5)
    n = 400
    close = pd.Series(100.0 + rng.normal(0, 1, n).cumsum())
    return {
        "open": close.shift(1).fillna(close.iloc[0]),
        "open_": close.shift(1).fillna(close.iloc[0]),
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": pd.Series(rng.integers(100, 5000, n).astype(float)),
    }


def _causal_symbols():
    out = []
    for name in sorted(vi._SYMBOL_TO_PACKAGE):
        func = getattr(vi, name)
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            continue
        if "causal" in sig.parameters:
            out.append(name)
    assert out, "expected at least the known pivot-family indicators"
    return out


@pytest.mark.parametrize("name", _causal_symbols())
def test_non_causal_mode_warns(name):
    func = getattr(vi, name)
    data = _data()
    sig = inspect.signature(func)
    kwargs = {}
    for p, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p in data:
            kwargs[p] = data[p]
    kwargs["causal"] = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func(**kwargs)
    assert any("causal=False" in str(w.message) for w in caught), (
        f"{name} exposes causal=False but does not warn — import "
        "warn_if_non_causal from v1indicators._causal and call it with the "
        "function name and the causal flag"
    )


@pytest.mark.parametrize("name", _causal_symbols())
def test_default_mode_is_silent(name):
    func = getattr(vi, name)
    data = _data()
    sig = inspect.signature(func)
    kwargs = {}
    for p, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p in data:
            kwargs[p] = data[p]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func(**kwargs)
    assert not caught, f"{name} emitted warnings under default settings: {caught}"
