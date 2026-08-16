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

    Requires a DatetimeIndex and computes running session bounds for each
    day. Times are interpreted in the index's own (naive or tz-aware) clock;
    there is no timezone conversion. ``start > end`` defines an overnight
    session that wraps midnight (e.g. ``"22:00", "04:00"``); bars after
    midnight belong to the session that started the previous calendar day,
    so the running high/low accumulate across the boundary. Bounds exist
    only while the session is active (NaN outside).
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

    if smin <= emin:
        active = (minutes >= smin) & (minutes < emin)
        day_key = pd.Series(idx.date, index=idx)
    else:
        # Overnight session: active on both sides of midnight. Post-midnight
        # bars join the session that started the previous calendar day.
        active = (minutes >= smin) | (minutes < emin)
        dates = pd.Series(idx.date, index=idx)
        prev_dates = pd.Series((idx - pd.Timedelta(days=1)).date, index=idx)
        day_key = dates.where(~(active & (minutes < emin)), prev_dates)

    day_key = day_key.where(active)

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
