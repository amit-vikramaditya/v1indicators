import pandas as pd

from .._utils import check_series
from ...foundational.trend.long_run import long_run
from ...foundational.trend.short_run import short_run
from ...foundational.volume.obv import obv


def aobv(
    close: pd.Series,
    volume: pd.Series,
    fast: int = 4,
    slow: int = 12,
    run_length: int = 2,
    min_lookback: int = 2,
    max_lookback: int = 2,
) -> pd.DataFrame:
    """Archer OBV with MA states and rolling min/max."""
    if min(fast, slow, run_length, min_lookback, max_lookback) <= 0:
        raise ValueError("all periods must be > 0")

    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    obv_s = obv(close_s, volume_s)
    maf = obv_s.ewm(span=fast, adjust=False).mean()
    mas = obv_s.ewm(span=slow, adjust=False).mean()

    lr = long_run(maf, mas, length=run_length)
    sr = short_run(maf, mas, length=run_length)

    return pd.DataFrame(
        {
            "OBV": obv_s,
            f"OBV_min_{min_lookback}": obv_s.rolling(min_lookback).min(),
            f"OBV_max_{max_lookback}": obv_s.rolling(max_lookback).max(),
            f"OBVe_{fast}": maf,
            f"OBVe_{slow}": mas,
            f"AOBV_LR_{run_length}": lr,
            f"AOBV_SR_{run_length}": sr,
        }
    )
