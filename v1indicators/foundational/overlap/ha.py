import pandas as pd

from .._utils import check_series


def ha(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Heikin Ashi candles."""
    open_s = check_series(open_, "open")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    ha_close = (open_s + high_s + low_s + close_s) / 4.0
    ha_open = ((open_s.shift(1).fillna(open_s.iloc[0])) + (close_s.shift(1).fillna(close_s.iloc[0]))) / 2.0
    ha_high = pd.concat([high_s, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low_s, ha_open, ha_close], axis=1).min(axis=1)

    return pd.DataFrame({"HA_OPEN": ha_open, "HA_HIGH": ha_high, "HA_LOW": ha_low, "HA_CLOSE": ha_close})
