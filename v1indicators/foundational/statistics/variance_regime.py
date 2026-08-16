import numpy as np
import pandas as pd

from .._utils import check_series


def variance_regime(
    close: pd.Series,
    window: int = 10,
    quantile_window: int = 100,
) -> pd.Series:
    """Probability of the low-volatility variance regime.

    A two-state Gaussian likelihood-ratio regime detector (not a hidden
    Markov model — there is no state transition model). Rolling variance is
    compared against its own trailing 25th and 75th percentiles; the log
    likelihoods of the last ``window`` returns under each variance hypothesis
    are combined through a sigmoid into a [0, 1] probability of the
    LOW-variance state.

    p = sigmoid(LL_low - LL_high)

    Strictly causal: every quantity at bar i uses returns up to and
    including bar i, and percentiles of variance up to bar i.

    Parameters
    ----------
    close : pd.Series
        Price series (returns are computed internally).
    window : int
        Number of trailing returns scored under each hypothesis.
    quantile_window : int
        Trailing window for the variance percentiles.

    Returns
    -------
    pd.Series
        Regime probability named ``VARIANCE_REGIME`` in [0, 1]; high values
        indicate calm (low-variance) conditions, low values turbulence.
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    if quantile_window <= 1:
        raise ValueError("quantile_window must be > 1")

    close_s = check_series(close, "close")
    returns = close_s.pct_change()

    var = returns.rolling(window).var(ddof=0)
    var_low = var.rolling(quantile_window, min_periods=2).quantile(0.25)
    var_high = var.rolling(quantile_window, min_periods=2).quantile(0.75)

    # Enforce ordering and positivity of the two hypotheses.
    var_low = var_low.clip(lower=1e-12)
    var_high = var_high.clip(lower=var_low + 1e-12)

    sum_sq = (returns**2).rolling(window).sum()

    ll_low = -0.5 * window * np.log(2.0 * np.pi * var_low) - 0.5 * sum_sq / var_low
    ll_high = -0.5 * window * np.log(2.0 * np.pi * var_high) - 0.5 * sum_sq / var_high

    diff = (ll_low - ll_high).clip(-100.0, 100.0)
    out = 1.0 / (1.0 + np.exp(diff))
    out.name = "VARIANCE_REGIME"
    return out
