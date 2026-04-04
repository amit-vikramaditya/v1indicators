import numpy as np
import pandas as pd

from .._utils import check_series


def decay(signal: pd.Series, length: int = 5, mode: str = "linear") -> pd.Series:
    """Decay transformation of a trigger series."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if mode not in {"linear", "exponential"}:
        raise ValueError("mode must be 'linear' or 'exponential'")

    s = check_series(signal, "signal").fillna(0.0)
    trigger = (s > 0.0).astype(float)

    if mode == "linear":
        w = np.linspace(1.0, 0.0, length, endpoint=False)
        out = trigger.rolling(length).apply(lambda x: np.max(x * w[::-1]), raw=True)
    else:
        alpha = 2.0 / (length + 1.0)
        out = trigger.ewm(alpha=alpha, adjust=False).mean()

    out = out.fillna(0.0)
    out.name = f"DECAY_{mode}_{length}"
    return out
