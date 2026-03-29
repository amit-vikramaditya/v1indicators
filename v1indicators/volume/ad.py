import numpy as np
import pandas as pd

from .._utils import check_series


def ad(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    open_: pd.Series | None = None,
) -> pd.Series:
    """Accumulation/Distribution line (AD)."""
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    if open_ is not None:
        open_s = check_series(open_, "open")
        flow = (close_s - open_s)
        name = "ADo"
    else:
        flow = 2.0 * close_s - (high_s + low_s)
        name = "AD"

    hl = (high_s - low_s).replace(0.0, np.nan)
    out = (flow * (volume_s / hl)).fillna(0.0).cumsum()
    out.name = name
    return out
