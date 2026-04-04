import numpy as np
import pandas as pd


def _validate_step(step: int) -> int:
    step_i = int(step)
    if step_i <= 0:
        raise ValueError("step must be > 0")
    return step_i


def _step_groups(length: int, step: int) -> np.ndarray:
    step_i = _validate_step(step)
    return np.arange(length, dtype=np.int64) // step_i


def _expand_group_series(reduced: pd.Series, groups: np.ndarray, index: pd.Index, name: str | None = None) -> pd.Series:
    out = pd.Series(reduced.reindex(groups).to_numpy(), index=index)
    out.name = name if name is not None else reduced.name
    return out


def _resample_last(series: pd.Series, step: int) -> tuple[pd.Series, np.ndarray]:
    step_i = _validate_step(step)
    if step_i == 1:
        groups = np.arange(len(series), dtype=np.int64)
        return series.copy(), groups

    groups = _step_groups(len(series), step_i)
    reduced = series.groupby(groups).last()
    reduced.name = series.name
    return reduced, groups


def _resample_ohlc(
    open_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    close_s: pd.Series,
    step: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, np.ndarray]:
    step_i = _validate_step(step)
    if step_i == 1:
        groups = np.arange(len(close_s), dtype=np.int64)
        return open_s.copy(), high_s.copy(), low_s.copy(), close_s.copy(), groups

    groups = _step_groups(len(close_s), step_i)
    o = open_s.groupby(groups).first()
    h = high_s.groupby(groups).max()
    l = low_s.groupby(groups).min()
    c = close_s.groupby(groups).last()
    return o, h, l, c, groups


def _group_end_mask(length: int, step: int) -> np.ndarray:
    step_i = _validate_step(step)
    mask = np.zeros(length, dtype=bool)
    if length == 0:
        return mask
    end_idx = np.arange(step_i - 1, length, step_i, dtype=np.int64)
    if end_idx.size == 0 or end_idx[-1] != length - 1:
        end_idx = np.append(end_idx, length - 1)
    mask[end_idx] = True
    return mask
