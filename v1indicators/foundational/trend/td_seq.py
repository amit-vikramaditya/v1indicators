import pandas as pd

from .._utils import check_series


def td_seq(close: pd.Series, setup_length: int = 9) -> pd.DataFrame:
    """TD Sequential setup counts."""
    if setup_length <= 0:
        raise ValueError("setup_length must be > 0")

    close_s = check_series(close, "close")
    up = (close_s > close_s.shift(4)).astype(int)
    down = (close_s < close_s.shift(4)).astype(int)
    buy = up.groupby((up == 0).cumsum()).cumsum().clip(upper=setup_length)
    sell = down.groupby((down == 0).cumsum()).cumsum().clip(upper=setup_length)

    return pd.DataFrame({"TD_BUY": buy, "TD_SELL": sell})
