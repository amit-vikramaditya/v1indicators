import pandas as pd

from .._utils import check_series


def session_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    start: str = "08:30",
    end: str = "12:00",
) -> pd.DataFrame:
    """
    Intraday session high/low/range tracker.

    Requires DatetimeIndex and computes running session bounds for each day.
    """
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    if not isinstance(high_s.index, pd.DatetimeIndex):
        raise TypeError("high/low/close index must be a pandas DatetimeIndex")

    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    idx = high_s.index
    minutes = idx.hour * 60 + idx.minute
    smin = sh * 60 + sm
    emin = eh * 60 + em

    active = (minutes >= smin) & (minutes < emin)
    day_key = pd.Series(idx.date, index=idx).where(active)

    session_high = high_s.where(active).groupby(day_key).cummax()
    session_low = low_s.where(active).groupby(day_key).cummin()
    session_mid = (session_high + session_low) / 2.0

    touch_high = (close_s.shift(1) <= session_high.shift(1)) & (close_s >= session_high.shift(1))
    touch_low = (close_s.shift(1) >= session_low.shift(1)) & (close_s <= session_low.shift(1))

    return pd.DataFrame(
        {
            "SESSION_ACTIVE": pd.Series(active, index=idx),
            "SESSION_HIGH": session_high,
            "SESSION_LOW": session_low,
            "SESSION_MID": session_mid,
            "SESSION_RANGE": session_high - session_low,
            "TOUCH_SESSION_HIGH": touch_high.fillna(False),
            "TOUCH_SESSION_LOW": touch_low.fillna(False),
        }
    )
