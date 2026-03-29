import numpy as np
import pandas as pd
from numba import njit

from .._utils import check_series


@njit(cache=True)
def _hwma_kernel(price: np.ndarray, na: float, nb: float, nc: float) -> np.ndarray:
    m = price.shape[0]
    out = np.empty(m, dtype=np.float64)

    last_a = 0.0
    last_v = 0.0
    last_f = price[0]
    out[0] = price[0]

    for i in range(1, m):
        f = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * price[i]
        v = (1.0 - nb) * (last_v + last_a) + nb * (f - last_f)
        a = (1.0 - nc) * last_a + nc * (v - last_v)
        out[i] = f + v + 0.5 * a

        last_f = f
        last_v = v
        last_a = a

    return out


def hwma(close: pd.Series, na: float = 0.2, nb: float = 0.1, nc: float = 0.1) -> pd.Series:
    """Holt-Winter Moving Average."""
    if min(na, nb, nc) <= 0.0:
        raise ValueError("na, nb, and nc must be > 0")

    close_s = check_series(close, "close")
    arr = _hwma_kernel(close_s.to_numpy(dtype=np.float64), na, nb, nc)
    out = pd.Series(arr, index=close_s.index)
    out.name = "HWMA"
    return out
