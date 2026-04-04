import pandas as pd

from ..._utils import check_series


def thermo(
    high: pd.Series,
    low: pd.Series,
    length: int = 20,
    long: float = 2.0,
    short: float = 0.5,
) -> pd.DataFrame:
    """Elder's Thermometer."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if long <= 0 or short <= 0:
        raise ValueError("long and short must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")

    thermo_l = (low_s.shift(1) - low_s).abs()
    thermo_h = (high_s - high_s.shift(1)).abs()
    t = pd.concat([thermo_l, thermo_h], axis=1).max(axis=1)
    t_ma = t.ewm(span=length, adjust=False).mean()
    t_long = (t < (t_ma * long)).astype(int)
    t_short = (t > (t_ma * short)).astype(int)

    suffix = f"_{length}_{long}_{short}"
    return pd.DataFrame(
        {
            f"THERMO{suffix}": t,
            f"THERMOma{suffix}": t_ma,
            f"THERMOl{suffix}": t_long,
            f"THERMOs{suffix}": t_short,
        }
    )
