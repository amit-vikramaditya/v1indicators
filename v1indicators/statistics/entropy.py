import numpy as np
import pandas as pd

from .._utils import check_series


def entropy(close: pd.Series, length: int = 10, base: float = 2.0) -> pd.Series:
    """Shannon entropy of signs of returns over rolling window."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if base <= 0:
        raise ValueError("base must be > 0")

    close_s = check_series(close, "close")
    sign = (close_s.diff() > 0.0).astype(int)

    def _ent(x: np.ndarray) -> float:
        p = x.mean()
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        q = 1.0 - p
        return -(p * (np.log(p) / np.log(base)) + q * (np.log(q) / np.log(base)))

    out = sign.rolling(length).apply(_ent, raw=True)
    out.name = f"ENTROPY_{length}"
    return out
