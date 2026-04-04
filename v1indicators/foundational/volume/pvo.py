import numpy as np
import pandas as pd

from .._utils import check_series


def pvo(
    volume: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Percentage Volume Oscillator.

    PVO = 100 * (EMA(volume, fast) - EMA(volume, slow)) / EMA(volume, slow)
    """
    if min(fast, slow, signal) <= 0:
        raise ValueError("fast, slow, and signal must be > 0")

    volume_s = check_series(volume, "volume")

    ema_fast = volume_s.ewm(span=fast, adjust=False).mean()
    ema_slow = volume_s.ewm(span=slow, adjust=False).mean().replace(0.0, np.nan)
    pvo_line = 100.0 * (ema_fast - ema_slow) / ema_slow
    signal_line = pvo_line.ewm(span=signal, adjust=False).mean()
    hist = pvo_line - signal_line

    return pd.DataFrame(
        {
            "PVO": pvo_line,
            "PVO_SIGNAL": signal_line,
            "PVO_HIST": hist,
        }
    )
