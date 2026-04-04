import numpy as np
import pandas as pd

from ..._utils import check_series


def pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    method: str = "classic",
) -> pd.DataFrame:
    """
    Pivot point levels based on prior bar values.

    Supported methods: classic, fibonacci, woodie, camarilla.
    """
    method_l = method.lower()
    valid = {"classic", "fibonacci", "woodie", "camarilla"}
    if method_l not in valid:
        raise ValueError(f"method must be one of {sorted(valid)}")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    ph = high_s.shift(1)
    pl = low_s.shift(1)
    pc = close_s.shift(1)
    rng = ph - pl

    p = (ph + pl + pc) / 3.0
    r1 = pd.Series(np.nan, index=high_s.index)
    s1 = pd.Series(np.nan, index=high_s.index)
    r2 = pd.Series(np.nan, index=high_s.index)
    s2 = pd.Series(np.nan, index=high_s.index)
    r3 = pd.Series(np.nan, index=high_s.index)
    s3 = pd.Series(np.nan, index=high_s.index)
    r4 = pd.Series(np.nan, index=high_s.index)
    s4 = pd.Series(np.nan, index=high_s.index)

    if method_l == "classic":
        r1 = 2.0 * p - pl
        s1 = 2.0 * p - ph
        r2 = p + rng
        s2 = p - rng
        r3 = ph + 2.0 * (p - pl)
        s3 = pl - 2.0 * (ph - p)

    elif method_l == "fibonacci":
        r1 = p + 0.382 * rng
        s1 = p - 0.382 * rng
        r2 = p + 0.618 * rng
        s2 = p - 0.618 * rng
        r3 = p + 1.0 * rng
        s3 = p - 1.0 * rng

    elif method_l == "woodie":
        p = (ph + pl + 2.0 * pc) / 4.0
        r1 = 2.0 * p - pl
        s1 = 2.0 * p - ph
        r2 = p + rng
        s2 = p - rng
        r3 = ph + 2.0 * (p - pl)
        s3 = pl - 2.0 * (ph - p)

    elif method_l == "camarilla":
        r1 = pc + rng * (1.1 / 12.0)
        s1 = pc - rng * (1.1 / 12.0)
        r2 = pc + rng * (1.1 / 6.0)
        s2 = pc - rng * (1.1 / 6.0)
        r3 = pc + rng * (1.1 / 4.0)
        s3 = pc - rng * (1.1 / 4.0)
        r4 = pc + rng * (1.1 / 2.0)
        s4 = pc - rng * (1.1 / 2.0)

    return pd.DataFrame(
        {
            "PIVOT_P": p,
            "PIVOT_R1": r1,
            "PIVOT_S1": s1,
            "PIVOT_R2": r2,
            "PIVOT_S2": s2,
            "PIVOT_R3": r3,
            "PIVOT_S3": s3,
            "PIVOT_R4": r4,
            "PIVOT_S4": s4,
        }
    )
