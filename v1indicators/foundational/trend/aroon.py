import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .._utils import check_series


def _aroon_component(values: np.ndarray, length: int, use_max: bool) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)

    if values.shape[0] < length:
        return out

    windows = sliding_window_view(values, window_shape=length)
    valid = np.isfinite(windows).all(axis=1)

    aroon_vals = np.full(windows.shape[0], np.nan, dtype=np.float64)
    if np.any(valid):
        rev_windows = windows[valid, ::-1]
        if use_max:
            periods_since = np.argmax(rev_windows, axis=1).astype(np.float64)
        else:
            periods_since = np.argmin(rev_windows, axis=1).astype(np.float64)

        aroon_vals[valid] = 100.0 * (length - periods_since) / length

    out[length - 1 :] = aroon_vals
    return out


def aroon_up(high: pd.Series, length: int = 25) -> pd.Series:
    """Aroon Up line."""
    if length <= 0:
        raise ValueError("length must be > 0")

    high_s = check_series(high, "high")
    up = _aroon_component(high_s.to_numpy(dtype=np.float64), length, use_max=True)
    return pd.Series(up, index=high_s.index, name=f"AROON_UP_{length}")


def aroon_down(low: pd.Series, length: int = 25) -> pd.Series:
    """Aroon Down line."""
    if length <= 0:
        raise ValueError("length must be > 0")

    low_s = check_series(low, "low")
    down = _aroon_component(low_s.to_numpy(dtype=np.float64), length, use_max=False)
    return pd.Series(down, index=low_s.index, name=f"AROON_DOWN_{length}")


def aroon_osc(high: pd.Series, low: pd.Series, length: int = 25) -> pd.Series:
    """Aroon Oscillator = Aroon Up - Aroon Down."""
    up = aroon_up(high, length=length)
    down = aroon_down(low, length=length)
    osc = up - down
    osc.name = f"AROON_OSC_{length}"
    return osc


def aroon(high: pd.Series, low: pd.Series, length: int = 25) -> pd.Series:
    """Primary Aroon output as oscillator series."""
    return aroon_osc(high, low, length=length)
