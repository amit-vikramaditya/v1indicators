import pandas as pd

from .._utils import check_series


def efi(close: pd.Series, volume: pd.Series, length: int = 13, drift: int = 1) -> pd.Series:
    """Elder's Force Index."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if drift <= 0:
        raise ValueError("drift must be > 0")

    close_s = check_series(close, "close")
    volume_s = check_series(volume, "volume")

    raw = close_s.diff(drift) * volume_s
    out = raw.ewm(span=length, adjust=False).mean()
    out.name = f"EFI_{length}"
    return out
