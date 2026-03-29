import numpy as np
import pandas as pd

from .._utils import check_series


def ebsw(close: pd.Series, length: int = 40) -> pd.Series:
    """Even Better SineWave approximation."""
    if length <= 1:
        raise ValueError("length must be > 1")

    close_s = check_series(close, "close")
    hp = close_s - close_s.rolling(length).mean()
    phase = hp.rolling(length // 2 if length > 2 else 2).apply(
        lambda x: np.arctan2(x.iloc[-1], np.sqrt((x * x).sum() + 1e-12)), raw=False
    )
    out = np.sin(phase)
    out.name = f"EBSW_{length}"
    return out
