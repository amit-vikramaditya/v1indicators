import numpy as np
import pandas as pd

from .._utils import check_series


def ssf(close: pd.Series, length: int = 20, poles: int = 2) -> pd.Series:
    """Ehlers Super Smoother Filter (2-pole and 3-pole variants)."""
    if length <= 1:
        raise ValueError("length must be > 1")
    if poles not in (2, 3):
        raise ValueError("poles must be 2 or 3")

    close_s = check_series(close, "close")
    x = close_s.to_numpy(dtype=float)
    y = np.full_like(x, np.nan)

    a1 = np.exp(-np.sqrt(2.0) * np.pi / length)
    b1 = 2.0 * a1 * np.cos(np.sqrt(2.0) * np.pi / length)
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3

    if len(x) > 0:
        y[0] = x[0]
    if len(x) > 1:
        y[1] = x[1]

    for i in range(2, len(x)):
        y[i] = c1 * 0.5 * (x[i] + x[i - 1]) + c2 * y[i - 1] + c3 * y[i - 2]

    if poles == 3:
        z = np.full_like(y, np.nan)
        if len(y) > 0:
            z[0] = y[0]
        if len(y) > 1:
            z[1] = y[1]
        if len(y) > 2:
            z[2] = y[2]
        for i in range(3, len(y)):
            z[i] = c1 * 0.5 * (y[i] + y[i - 1]) + c2 * z[i - 1] + c3 * z[i - 2]
        y = z

    out = pd.Series(y, index=close_s.index, name=f"SSF_{length}_{poles}")
    return out
