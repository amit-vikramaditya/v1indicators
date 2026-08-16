import pandas as pd

from .._utils import check_series


def day_week_month_levels(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> pd.DataFrame:
    """
    Prior day/week/month key levels.

    Returns the COMPLETED prior period's open/high/low aligned to each bar:
    each bar sees yesterday's / last week's / last month's values, never the
    in-progress period. Calendar periods with no data (weekends, holidays,
    empty months) are skipped, so "prior day" means the prior TRADING day.
    Requires a DatetimeIndex.
    """
    open_s = check_series(open_, "open_")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")

    if not isinstance(open_s.index, pd.DatetimeIndex):
        raise TypeError("open/high/low index must be a pandas DatetimeIndex")

    df = pd.DataFrame({"open": open_s, "high": high_s, "low": low_s})

    agg = {"open": "first", "high": "max", "low": "min"}
    d = df.resample("1D").agg(agg)
    w = df.resample("1W").agg(agg)
    # Month-START labels: a bar in month M must map to the M-1 bin after
    # shift(1). Month-END labels are off by one (a March bar would map to
    # the February label whose shifted value is January's aggregate).
    m = df.resample("1MS").agg(agg)

    def _prior_period(levels: pd.DataFrame) -> pd.DataFrame:
        # Calendar bins without data (weekends, holidays, empty months) must
        # be dropped BEFORE shift(1): otherwise the shift lands on an empty
        # bin and every bar of the following period gets NaN levels.
        return levels.dropna(how="any").shift(1).reindex(df.index, method="ffill")

    d_prev = _prior_period(d)
    w_prev = _prior_period(w)
    m_prev = _prior_period(m)

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
