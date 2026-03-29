import pandas as pd

from .._utils import check_series


def equal_highs_lows(
    high: pd.Series,
    low: pd.Series,
    length: int = 3,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Equal highs/lows detector.

    A level is considered equal when two consecutive local pivots are within
    `threshold` percent of each other.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")

    window = 2 * length + 1
    pivot_high = high_s.where(high_s == high_s.rolling(window).max().shift(-length))
    pivot_low = low_s.where(low_s == low_s.rolling(window).min().shift(-length))

    last_ph = pivot_high.ffill()
    prev_ph = last_ph.shift(1)
    last_pl = pivot_low.ffill()
    prev_pl = last_pl.shift(1)

    eqh = pivot_high.notna() & (((last_ph - prev_ph).abs() / prev_ph.abs().replace(0.0, pd.NA)) <= threshold)
    eql = pivot_low.notna() & (((last_pl - prev_pl).abs() / prev_pl.abs().replace(0.0, pd.NA)) <= threshold)

    return pd.DataFrame(
        {
            "PIVOT_HIGH": pivot_high,
            "PIVOT_LOW": pivot_low,
            "EQUAL_HIGH": eqh.fillna(False),
            "EQUAL_LOW": eql.fillna(False),
            "LAST_HIGH": last_ph,
            "LAST_LOW": last_pl,
        }
    )
