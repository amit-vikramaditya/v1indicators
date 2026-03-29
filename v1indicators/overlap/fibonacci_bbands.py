import pandas as pd

from .._utils import check_series
from ..statistics.stdev import stdev
from .hlc3 import hlc3
from .vwma import vwma


_FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.764, 1.0)


def fibonacci_bbands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 200,
    mult: float = 3.0,
    source: pd.Series | None = None,
) -> pd.DataFrame:
    """Fibonacci Bollinger Bands with VWMA basis.

    Formula:
        basis = VWMA(source, volume, length)
        dev = mult * STDEV(source, length)
        upper_k = basis + ratio_k * dev
        lower_k = basis - ratio_k * dev

    If ``source`` is not provided, ``HLC3`` is used.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if mult <= 0:
        raise ValueError("mult must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    src = hlc3(high_s, low_s, close_s) if source is None else check_series(source, "source")
    basis = vwma(src, volume_s, length=length)
    dev = stdev(src, length=length) * mult

    out = {
        "FBB_BASIS": basis,
    }

    for ratio in _FIB_RATIOS:
        tag = int(round(ratio * 1000))
        out[f"FBB_UPPER_{tag}"] = basis + ratio * dev
        out[f"FBB_LOWER_{tag}"] = basis - ratio * dev

    return pd.DataFrame(out, index=close_s.index)
