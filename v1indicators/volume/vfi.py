import numpy as np
import pandas as pd

from .._utils import check_series


def vfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 130,
    coef: float = 0.2,
    vcoef: float = 2.5,
    signal: int = 5,
    smooth: bool = False,
) -> pd.DataFrame:
    """
    Volume Flow Indicator (VFI).

    Returns VFI line, signal line, and histogram.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if signal <= 0:
        raise ValueError("signal must be > 0")
    if coef < 0:
        raise ValueError("coef must be >= 0")
    if vcoef <= 0:
        raise ValueError("vcoef must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    typical = (high_s + low_s + close_s) / 3.0

    safe_typical = typical.where(typical > 0.0)
    inter = np.log(safe_typical) - np.log(safe_typical.shift(1))
    vinter = inter.rolling(30).std(ddof=0)
    cutoff = coef * vinter * close_s

    vave = volume_s.rolling(length).mean().shift(1)
    vmax = vave * vcoef
    vc = volume_s.where(volume_s < vmax, vmax)

    mf = typical - typical.shift(1)
    vcp = pd.Series(
        np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0.0)),
        index=close_s.index,
        dtype=np.float64,
    )

    vfi_line = vcp.rolling(length).sum() / vave.replace(0.0, np.nan)
    if smooth:
        vfi_line = vfi_line.rolling(3).mean()

    signal_line = vfi_line.ewm(span=signal, adjust=False).mean()
    hist = vfi_line - signal_line

    return pd.DataFrame(
        {
            "VFI": vfi_line,
            "VFI_SIGNAL": signal_line,
            "VFI_HIST": hist,
        }
    )
