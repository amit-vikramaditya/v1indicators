import warnings

import pandas as pd

from .._utils import check_series


def dpo(close: pd.Series, length: int = 20) -> pd.Series:
    """Detrended Price Oscillator.

    .. deprecated:: 1.0.0
       The DPO displaces price BACKWARD by ``length // 2 + 1`` bars by
       definition, so its values at bar i are only knowable with future
       data. It is a visual detrending aid, not a causal signal, and is
       retained only for backward compatibility.
    """
    warnings.warn(
        "dpo is deprecated: it is look-ahead by definition (price displaced "
        "backward); use only for visual detrending, never for signals.",
        DeprecationWarning,
        stacklevel=2,
    )
    if length <= 0:
        raise ValueError("length must be > 0")

    close_s = check_series(close, "close")
    shift = length // 2 + 1
    sma = close_s.rolling(length).mean()
    out = close_s.shift(-shift) - sma
    out.name = f"DPO_{length}"
    return out
