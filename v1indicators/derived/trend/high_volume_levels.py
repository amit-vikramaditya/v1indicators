import pandas as pd

from .._utils import check_series
from ...foundational.volume.delta_volume import delta_volume


def high_volume_levels(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
    vol_length: int = 2,
    causal: bool = True,
) -> pd.DataFrame:
    """
    Support/resistance levels confirmed by extreme directional volume.

    Uses delta volume with rolling high/low thresholds and pivot detection.

    When ``causal=True`` (default), pivots and every derived level/signal are
    delayed to the bar where the pivot is actually confirmed (pivot bar plus
    ``lookback`` bars), so each output value was knowable in real time.
    ``causal=False`` restores the legacy retrospective placement, suitable
    only for plotting historical structure and not for backtests.
    """

    if lookback <= 0 or vol_length <= 0:
        raise ValueError("lookback and vol_length must be > 0")

    open_s = check_series(open_, "open_")
    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    dvol = delta_volume(open_s, close_s, volume_s)["DELTA_VOLUME"] / 2.5
    vol_hi = dvol.rolling(vol_length).max()
    vol_lo = dvol.rolling(vol_length).min()

    ph = high_s.where(high_s == high_s.rolling(2 * lookback + 1).max().shift(-lookback))
    pl = low_s.where(low_s == low_s.rolling(2 * lookback + 1).min().shift(-lookback))

    ph = ph.where(dvol < vol_lo)
    pl = pl.where(dvol > vol_hi)

    if causal:
        ph = ph.shift(lookback)
        pl = pl.shift(lookback)

    resistance = ph.ffill()
    support = pl.ffill()

    break_resistance = (close_s.shift(1) <= resistance.shift(1)) & (close_s > resistance.shift(1))
    break_support = (close_s.shift(1) >= support.shift(1)) & (close_s < support.shift(1))

    return pd.DataFrame(
        {
            "HV_PIVOT_HIGH": ph,
            "HV_PIVOT_LOW": pl,
            "HV_RESISTANCE": resistance,
            "HV_SUPPORT": support,
            "HV_BREAK_RESISTANCE": break_resistance.fillna(False),
            "HV_BREAK_SUPPORT": break_support.fillna(False),
        }
    )
