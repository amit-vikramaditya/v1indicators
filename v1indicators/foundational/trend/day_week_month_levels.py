import pandas as pd

from .._utils import check_series


def day_week_month_levels(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> pd.DataFrame:
    """
    Prior day/week/month key levels.

    Returns previous period open/high/low levels aligned to each bar.
    Requires a DatetimeIndex.
    """
    open_s = check_series(open_, "open_")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")

    if not isinstance(open_s.index, pd.DatetimeIndex):
        raise TypeError("open/high/low index must be a pandas DatetimeIndex")

    df = pd.DataFrame({"open": open_s, "high": high_s, "low": low_s})

    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min"})
    w = df.resample("1W").agg({"open": "first", "high": "max", "low": "min"})
    m = df.resample("1ME").agg({"open": "first", "high": "max", "low": "min"})

    d_prev = d.shift(1).reindex(df.index, method="ffill")
    w_prev = w.shift(1).reindex(df.index, method="ffill")
    m_prev = m.shift(1).reindex(df.index, method="ffill")

    return pd.DataFrame(
        {
            "PD_OPEN": d_prev["open"],
            "PD_HIGH": d_prev["high"],
            "PD_LOW": d_prev["low"],
            "PW_OPEN": w_prev["open"],
            "PW_HIGH": w_prev["high"],
            "PW_LOW": w_prev["low"],
            "PM_OPEN": m_prev["open"],
            "PM_HIGH": m_prev["high"],
            "PM_LOW": m_prev["low"],
        },
        index=df.index,
    )
