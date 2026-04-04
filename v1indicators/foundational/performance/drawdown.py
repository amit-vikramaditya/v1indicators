import numpy as np
import pandas as pd

from ..._utils import check_series


def drawdown(close: pd.Series) -> pd.DataFrame:
    """
    Drawdown metrics from cumulative peak.

    Returns absolute drawdown, percentage drawdown, and log drawdown.
    """
    close_s = check_series(close, "close")
    peak = close_s.cummax().replace(0.0, np.nan)

    dd_abs = close_s - peak
    dd_pct = (close_s / peak) - 1.0
    dd_log = np.log(close_s.replace(0.0, np.nan) / peak)

    return pd.DataFrame(
        {
            "DRAWDOWN": dd_abs,
            "DRAWDOWN_PCT": dd_pct,
            "DRAWDOWN_LOG": dd_log,
        }
    )
