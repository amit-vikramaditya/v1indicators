import pandas as pd

from .._utils import check_series
from ..overlap.hwma import hwma


def hwc(
    close: pd.Series,
    scalar: float = 1.0,
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    nd: float = 0.1,
) -> pd.DataFrame:
    """Holt-Winter Channel."""
    if min(scalar, na, nb, nc, nd) <= 0:
        raise ValueError("all parameters must be > 0")

    close_s = check_series(close, "close")
    mid = hwma(close_s, na=na, nb=nb, nc=nc)
    var = (close_s - mid).pow(2).ewm(alpha=nd, adjust=False).mean()
    std = var.pow(0.5)

    upper = mid + scalar * std
    lower = mid - scalar * std
    width = upper - lower
    pct = (close_s - lower) / width.replace(0.0, pd.NA)

    return pd.DataFrame({"HWM_1": mid, "HWL_1": lower, "HWU_1": upper, "HWW_1": width, "HWPCT_1": pct})
