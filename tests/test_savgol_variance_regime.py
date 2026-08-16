"""Tests for the causal Savitzky-Golay smoother and variance regime detector."""

import numpy as np
import pandas as pd
import pytest

from v1indicators import savgol, variance_regime
from v1indicators.foundational.overlap.savgol import savgol_weights


def test_savgol_weights_match_reference_quadratic_sets():
    # Parity with the production-hardcoded quadratic endpoint weight sets the
    # port was derived from (1vcapital): w=5 and w=7, polyorder=2.
    w5 = [-0.08571429, 0.34285714, 0.48571429, 0.34285714, -0.08571429]
    w7 = [-0.0952381, 0.14285714, 0.28571429, 0.33333333,
          0.28571429, 0.14285714, -0.0952381]
    assert np.allclose(savgol_weights(5, 2), w5, atol=1e-7)
    assert np.allclose(savgol_weights(7, 2), w7, atol=1e-7)


def test_savgol_weights_partition_unity():
    # An LS polynomial fit reproduces constants exactly: weights sum to 1.
    for window in (5, 7, 9, 12):
        for order in (1, 2, 3):
            if window <= order:
                continue
            assert np.isclose(savgol_weights(window, order).sum(), 1.0)


def test_savgol_reproduces_polynomials_at_the_delayed_position():
    # The trailing-window centred fit equals the classic centred SG value
    # delayed by (window-1)/2 bars: for window=7 the delay is 3.
    lin = pd.Series(np.arange(50, dtype=float))
    quad = pd.Series(np.arange(50, dtype=float) ** 2)
    sm_lin = savgol(lin, window=7, polyorder=2)
    sm_quad = savgol(quad, window=7, polyorder=2)
    assert np.allclose(sm_lin.to_numpy()[6:], lin.shift(3).to_numpy()[6:])
    assert np.allclose(sm_quad.to_numpy()[6:], quad.shift(3).to_numpy()[6:])
    assert sm_lin.name == "SAVGOL_7"


def test_savgol_warmup_and_validation():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = savgol(x, window=5, polyorder=2)
    assert np.isnan(out.iloc[:4]).all()
    assert not np.isnan(out.iloc[4])
    with pytest.raises(ValueError):
        savgol(x, window=2, polyorder=2)
    with pytest.raises(ValueError):
        savgol_weights(3, -1)


def test_variance_regime_transitions():
    rng = np.random.default_rng(3)
    calm_then_wild = list(rng.normal(0.0, 0.002, 300)) + list(rng.normal(0.0, 0.05, 100))
    wild_then_calm = list(rng.normal(0.0, 0.05, 300)) + list(rng.normal(0.0, 0.002, 100))

    def regime(returns):
        prices = 100.0 * pd.Series(np.cumprod(1.0 + np.array(returns)))
        return variance_regime(prices, window=10, quantile_window=100)

    r1 = regime(calm_then_wild)
    r2 = regime(wild_then_calm)
    # After the calm -> turbulent transition the low-variance probability
    # must be LOW; after turbulent -> calm it must be HIGH.
    assert r1.iloc[-1] < 0.5
    assert r2.iloc[-1] > 0.5
    # Bounds
    for r in (r1, r2):
        assert (r.dropna() >= 0.0).all() and (r.dropna() <= 1.0).all()
    assert r2.name == "VARIANCE_REGIME"


def test_variance_regime_input_validation():
    x = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        variance_regime(x, window=0)
    with pytest.raises(ValueError):
        variance_regime(x, window=10, quantile_window=1)
