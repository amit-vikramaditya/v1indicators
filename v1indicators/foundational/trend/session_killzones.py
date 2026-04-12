import pandas as pd

from .._utils import check_series


def _session_mask(index: pd.DatetimeIndex, start_hhmm: str, end_hhmm: str) -> pd.Series:
    sh, sm = map(int, start_hhmm.split(":"))
    eh, em = map(int, end_hhmm.split(":"))
    minutes = index.hour * 60 + index.minute
    start_m = sh * 60 + sm
    end_m = eh * 60 + em
    if start_m <= end_m:
        mask = (minutes >= start_m) & (minutes < end_m)
    else:
        mask = (minutes >= start_m) | (minutes < end_m)
    return pd.Series(mask, index=index)


def session_killzones(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    asia: tuple[str, str] = ("20:00", "00:00"),
    london: tuple[str, str] = ("02:00", "05:00"),
    ny_am: tuple[str, str] = ("09:30", "11:00"),
) -> pd.DataFrame:
    """
    Session killzone range tracker.

    Computes active-session high/low and range for Asia, London, and NY AM windows.
    Requires a DatetimeIndex.
    """
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    if not isinstance(high_s.index, pd.DatetimeIndex):
        raise TypeError("high/low/close index must be a pandas DatetimeIndex")

    idx = high_s.index

    def _zone(prefix: str, session: tuple[str, str]):
        mask = _session_mask(idx, session[0], session[1])
        zone_day = pd.Series(idx.date, index=idx).where(mask)
        z_high = high_s.where(mask).groupby(zone_day).cummax()
        z_low = low_s.where(mask).groupby(zone_day).cummin()
        z_range = z_high - z_low
        return {
            f"{prefix}_ACTIVE": mask,
            f"{prefix}_HIGH": z_high,
            f"{prefix}_LOW": z_low,
            f"{prefix}_RANGE": z_range,
            f"{prefix}_MID": (z_high + z_low) / 2.0,
        }

    data = {}
    data.update(_zone("ASIA", asia))
    data.update(_zone("LONDON", london))
    data.update(_zone("NY_AM", ny_am))
    data["CLOSE"] = close_s

    return pd.DataFrame(data, index=idx)
