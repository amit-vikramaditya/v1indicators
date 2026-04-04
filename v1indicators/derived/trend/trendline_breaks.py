import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series
from ...foundational.volatility.atr import atr


@njit
def _trendline_breaks_kernel(
    close_v: np.ndarray,
    pivot_high_v: np.ndarray,
    pivot_low_v: np.ndarray,
    slope_v: np.ndarray,
    length: int,
):
    n = close_v.shape[0]

    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    slope_upper = np.full(n, np.nan, dtype=np.float64)
    slope_lower = np.full(n, np.nan, dtype=np.float64)
    breakout_up = np.zeros(n, dtype=np.bool_)
    breakout_down = np.zeros(n, dtype=np.bool_)

    cur_upper = np.nan
    cur_lower = np.nan
    cur_slope_upper = 0.0
    cur_slope_lower = 0.0

    prev_dyn_upper = np.nan
    prev_dyn_lower = np.nan

    for i in range(n):
        ph = pivot_high_v[i]
        pl = pivot_low_v[i]
        s = slope_v[i]

        if not np.isnan(ph):
            cur_upper = ph
            if not np.isnan(s):
                cur_slope_upper = s
        elif not np.isnan(cur_upper):
            cur_upper = cur_upper - cur_slope_upper

        if not np.isnan(pl):
            cur_lower = pl
            if not np.isnan(s):
                cur_slope_lower = s
        elif not np.isnan(cur_lower):
            cur_lower = cur_lower + cur_slope_lower

        upper[i] = cur_upper
        lower[i] = cur_lower
        slope_upper[i] = cur_slope_upper
        slope_lower[i] = cur_slope_lower

        dyn_upper = np.nan
        dyn_lower = np.nan

        if not np.isnan(cur_upper):
            dyn_upper = cur_upper - cur_slope_upper * length
        if not np.isnan(cur_lower):
            dyn_lower = cur_lower + cur_slope_lower * length

        if i > 0 and not np.isnan(close_v[i]) and not np.isnan(close_v[i - 1]):
            if (
                not np.isnan(dyn_upper)
                and not np.isnan(prev_dyn_upper)
                and close_v[i] > dyn_upper
                and close_v[i - 1] <= prev_dyn_upper
            ):
                breakout_up[i] = True

            if (
                not np.isnan(dyn_lower)
                and not np.isnan(prev_dyn_lower)
                and close_v[i] < dyn_lower
                and close_v[i - 1] >= prev_dyn_lower
            ):
                breakout_down[i] = True

        prev_dyn_upper = dyn_upper
        prev_dyn_lower = dyn_lower

    return upper, lower, slope_upper, slope_lower, breakout_up, breakout_down


def trendline_breaks(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    mult: float = 1.0,
    slope_method: str = "atr",
) -> pd.DataFrame:
    """
    Pivot-based trendlines with breakout flags.

    Creates descending resistance and ascending support trendlines from pivots,
    then flags close breakouts through those lines.

    This is a retrospective structure indicator: pivot anchors are only
    confirmed after `length` future bars.
    """
    if length <= 1:
        raise ValueError("length must be > 1")
    if mult < 0:
        raise ValueError("mult must be >= 0")

    method = slope_method.lower()
    valid_methods = {"atr", "stdev", "linreg"}
    if method not in valid_methods:
        raise ValueError(f"slope_method must be one of {sorted(valid_methods)}")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    if method == "atr":
        slope = atr(high_s, low_s, close_s, length=length) / float(length)
        slope = slope * mult
    elif method == "stdev":
        slope = close_s.rolling(length).std(ddof=0) / float(length)
        slope = slope * mult
    else:
        n = pd.Series(np.arange(close_s.size, dtype=np.float64), index=close_s.index)
        mean_src = close_s.rolling(length).mean()
        mean_n = n.rolling(length).mean()
        mean_srcn = (close_s * n).rolling(length).mean()
        var_n = n.rolling(length).var(ddof=0).replace(0.0, np.nan)
        slope = (mean_srcn - mean_src * mean_n).abs() / var_n / 2.0
        slope = slope * mult

    window = 2 * length + 1
    pivot_high = high_s.where(high_s == high_s.rolling(window).max().shift(-length))
    pivot_low = low_s.where(low_s == low_s.rolling(window).min().shift(-length))

    upper, lower, slope_upper, slope_lower, breakout_up, breakout_down = _trendline_breaks_kernel(
        close_s.to_numpy(dtype=np.float64),
        pivot_high.to_numpy(dtype=np.float64),
        pivot_low.to_numpy(dtype=np.float64),
        slope.to_numpy(dtype=np.float64),
        int(length),
    )

    return pd.DataFrame(
        {
            "TRENDLINE_UPPER": upper,
            "TRENDLINE_LOWER": lower,
            "TRENDLINE_SLOPE_UPPER": slope_upper,
            "TRENDLINE_SLOPE_LOWER": slope_lower,
            "BREAKOUT_UP": breakout_up,
            "BREAKOUT_DOWN": breakout_down,
        },
        index=close_s.index,
    )
