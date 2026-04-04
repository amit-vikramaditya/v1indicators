import pandas as pd

from .._utils import check_series


def amat(close: pd.Series, fast: int = 8, slow: int = 21) -> pd.DataFrame:
    """Archer MA trend states from EMA crossover."""
    if fast <= 0 or slow <= 0:
        raise ValueError("fast and slow must be > 0")

    close_s = check_series(close, "close")
    ema_fast = close_s.ewm(span=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False).mean()

    long_state = (ema_fast > ema_slow).astype(int)
    short_state = (ema_fast < ema_slow).astype(int)

    return pd.DataFrame({f"AMATe_{fast}_{slow}_L": long_state, f"AMATe_{fast}_{slow}_S": short_state})
