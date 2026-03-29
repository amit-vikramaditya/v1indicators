import pandas as pd

from .._utils import check_series


def smi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    fast: int = 3,
    slow: int = 3,
    signal: int = 3,
) -> pd.DataFrame:
    """Stochastic Momentum Index."""
    if min(length, fast, slow, signal) <= 0:
        raise ValueError("all periods must be > 0")

    high_s = check_series(high, "high")
    low_s = check_series(low, "low")
    close_s = check_series(close, "close")

    hh = high_s.rolling(length).max()
    ll = low_s.rolling(length).min()
    m = close_s - (hh + ll) / 2.0
    d = (hh - ll) / 2.0

    sm = m.ewm(span=fast, adjust=False).mean().ewm(span=slow, adjust=False).mean()
    sd = d.ewm(span=fast, adjust=False).mean().ewm(span=slow, adjust=False).mean()
    smi_line = 100.0 * sm / sd.replace(0.0, pd.NA)
    sig = smi_line.ewm(span=signal, adjust=False).mean()

    return pd.DataFrame({"SMI": smi_line, "SMI_SIGNAL": sig})
