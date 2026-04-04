import numpy as np
import pandas as pd

from .._utils import check_series


def ppo(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Percentage Price Oscillator (PPO).

    PPO = 100 * (EMA_fast - EMA_slow) / EMA_slow
    """
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be > 0")
    if fast >= slow:
        raise ValueError("fast must be < slow")

    close_s = check_series(close, "close")
    ema_fast = close_s.ewm(span=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False).mean().replace(0.0, np.nan)

    result = 100.0 * (ema_fast - ema_slow) / ema_slow
    result.name = f"PPO_{fast}_{slow}"
    return result
