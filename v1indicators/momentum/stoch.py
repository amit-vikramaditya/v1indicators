import pandas as pd

from .._utils import check_series
from .stochastic import stochastic


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    smooth: int = 3,
    k: int | None = None,
    d: int | None = None,
    smooth_k: int | None = None,
) -> pd.DataFrame:
    """Alias for stochastic oscillator with compatibility parameter aliases."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    length_eff = int(k) if k is not None else int(length)
    smooth_eff = int(smooth_k) if smooth_k is not None else int(smooth)
    d_eff = int(d) if d is not None else smooth_eff

    if min(length_eff, smooth_eff, d_eff) <= 0:
        raise ValueError("length/smooth parameters must be > 0")

    out = stochastic(high_s, low_s, close_s, length=length_eff, smooth=smooth_eff)
    if d_eff != smooth_eff:
        out["STOCH_D"] = out["STOCH_K"].rolling(d_eff).mean()
    return out
