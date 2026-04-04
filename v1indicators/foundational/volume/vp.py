import numpy as np
import pandas as pd

from .._utils import check_series


def vp(close: pd.Series, volume: pd.Series, width: int = 10, sort: bool = False) -> pd.DataFrame:
    """Volume Profile snapshot by price bins."""
    if width <= 0:
        raise ValueError("width must be > 0")

    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    df = pd.DataFrame({"close": close_s, "volume": volume_s})
    if sort:
        bins = pd.cut(df["close"], bins=width, include_lowest=True)
        grp = df.groupby(bins, observed=False)
        out = pd.DataFrame(
            {
                "low_close": grp["close"].min(),
                "mean_close": grp["close"].mean(),
                "high_close": grp["close"].max(),
                "total_volume": grp["volume"].sum(),
            }
        ).reset_index(drop=True)
    else:
        idx_chunks = np.array_split(np.arange(len(df)), width)
        out = pd.DataFrame(
            [
                {
                    **({} if len(ix) == 0 else {
                        "low_close": df.iloc[ix]["close"].min(),
                        "mean_close": df.iloc[ix]["close"].mean(),
                        "high_close": df.iloc[ix]["close"].max(),
                        "total_volume": df.iloc[ix]["volume"].sum(),
                    })
                }
                for ix in idx_chunks
            ]
        )

    out.name = f"VP_{width}"
    return out
