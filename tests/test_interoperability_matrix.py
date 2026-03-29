import inspect
from numbers import Real
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import v1indicators as vi


def _build_scenario(name: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n, freq="15min")

    if name == "trend_up":
        base = 100.0 + np.linspace(0.0, 50.0, n) + rng.normal(0.0, 0.8, n).cumsum() * 0.1
    elif name == "trend_down":
        base = 150.0 - np.linspace(0.0, 60.0, n) + rng.normal(0.0, 0.9, n).cumsum() * 0.12
    elif name == "sideways":
        base = 100.0 + rng.normal(0.0, 1.2, n).cumsum() * 0.05
    elif name == "volatile":
        base = 100.0 + rng.normal(0.0, 2.5, n).cumsum() * 0.25
    elif name == "gappy":
        base = 110.0 + rng.normal(0.0, 1.0, n).cumsum() * 0.12
        jumps = np.zeros(n)
        jump_idx = np.arange(40, n, 70)
        jumps[jump_idx] = rng.normal(0.0, 8.0, len(jump_idx))
        base = base + np.cumsum(jumps)
    elif name == "flat":
        base = np.full(n, 100.0) + rng.normal(0.0, 1e-5, n)
    elif name == "low_volume":
        base = 90.0 + np.linspace(0.0, 20.0, n) + rng.normal(0.0, 0.4, n).cumsum() * 0.06
    elif name == "nan_streaks":
        base = 105.0 + rng.normal(0.0, 1.0, n).cumsum() * 0.1
    else:
        raise ValueError(f"unknown scenario {name}")

    close = pd.Series(base, index=index, dtype=np.float64)
    open_ = close.shift(1).fillna(close.iloc[0])
    wick_up = np.abs(rng.normal(0.5, 0.15, n))
    wick_dn = np.abs(rng.normal(0.5, 0.15, n))

    high = pd.Series(np.maximum(open_.to_numpy(), close.to_numpy()) + wick_up, index=index, dtype=np.float64)
    low = pd.Series(np.minimum(open_.to_numpy(), close.to_numpy()) - wick_dn, index=index, dtype=np.float64)

    if name == "low_volume":
        volume = pd.Series(rng.integers(1, 30, n), index=index, dtype=np.float64)
    else:
        volume = pd.Series(rng.integers(100, 2000, n), index=index, dtype=np.float64)

    if name == "nan_streaks":
        for start in (80, 210, 430):
            end = min(start + 8, n)
            close.iloc[start:end] = np.nan
            open_.iloc[start:end] = np.nan
            high.iloc[start:end] = np.nan
            low.iloc[start:end] = np.nan
            volume.iloc[start:end] = np.nan

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _param_value(name: str, data: pd.DataFrame, annotation: Any) -> Any:
    key = name.lower()
    ann_name = str(annotation).lower()
    expects_series = "series" in ann_name or annotation is pd.Series

    if key in {"open", "open_", "o"} or key.startswith("open"):
        return data["open"]
    if key in {"high", "h"} or key.startswith("high"):
        return data["high"]
    if key in {"low", "l"} or key.startswith("low"):
        return data["low"]
    if key in {"close", "c"} or key.startswith("close"):
        return data["close"]
    if key in {"volume", "vol", "v"} or key.startswith("volume"):
        return data["volume"]

    if key in {"src", "source", "series", "data", "price", "input", "x"}:
        return data["close"]

    if key in {"df", "ohlcv", "dataframe"}:
        return data.copy()

    if key == "fast":
        return data["close"] if expects_series else 12
    if key == "slow":
        return data["close"] if expects_series else 26
    if key == "signal":
        return data["close"] if expects_series else 9

    if key == "short":
        return 7
    if key in {"medium", "mid"}:
        return 14
    if key == "long":
        return 28

    if key in {"length", "period", "window", "lookback", "n", "k", "d", "left", "right"}:
        return 14

    if "length" in key or "period" in key or "lookback" in key or key.endswith("_len"):
        return 14

    if key in {"mult", "multiplier", "factor", "threshold", "alpha", "beta"}:
        return 2.0
    if key == "q":
        return 0.5

    if key in {"drift", "offset", "step", "stride", "horizon", "neighbors_count", "max_bars_back"}:
        if key == "max_bars_back":
            return 200
        return 4

    if key in {"mamode", "ma_type", "mode", "method", "preset"}:
        if key == "preset":
            return "default"
        return "ema"

    if key.startswith("use_") or key.startswith("show_"):
        return True

    raise KeyError(name)


def _build_kwargs(func: Callable[..., Any], data: pd.DataFrame) -> dict[str, Any]:
    signature = inspect.signature(func)
    kwargs: dict[str, Any] = {}

    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        if parameter.default is inspect._empty:
            try:
                kwargs[parameter.name] = _param_value(parameter.name, data, parameter.annotation)
            except KeyError:
                annotation = parameter.annotation
                if annotation is int:
                    kwargs[parameter.name] = 14
                elif annotation is float:
                    kwargs[parameter.name] = 1.0
                elif annotation is bool:
                    kwargs[parameter.name] = True
                elif annotation is str:
                    kwargs[parameter.name] = "ema"
                else:
                    raise ValueError(f"unsupported required parameter: {parameter.name}")

    return kwargs


def _normalize_output(name: str, out: Any, index: pd.Index) -> tuple[pd.DataFrame, str]:
    if isinstance(out, pd.Series):
        series = out.copy()
        series.name = f"{name}__{series.name or 'value'}"
        kind = "timeseries" if len(series) == len(index) else "profile"
        return series.to_frame(), kind

    if isinstance(out, pd.DataFrame):
        frame = out.copy()
        frame.columns = [f"{name}__{column}" for column in frame.columns]
        kind = "timeseries" if len(frame) == len(index) else "profile"
        return frame, kind

    if isinstance(out, dict):
        numeric_items = {k: v for k, v in out.items() if isinstance(v, Real)}
        series_items = {k: v for k, v in out.items() if isinstance(v, pd.Series)}

        if numeric_items and not series_items:
            frame = pd.DataFrame({f"{name}__{k}": [float(v)] for k, v in numeric_items.items()})
            return frame, "levels"

        if series_items:
            parts: list[pd.DataFrame] = []
            for key, value in series_items.items():
                series = value.copy()
                series.name = f"{name}__{key}"
                parts.append(series.to_frame())
            frame = pd.concat(parts, axis=1)
            kind = "timeseries" if len(frame) == len(index) else "profile"
            return frame, kind

        raise TypeError("unsupported dict output")

    if isinstance(out, np.ndarray) and out.ndim == 1 and len(out) == len(index):
        return pd.DataFrame({f"{name}__array": out}, index=index), "timeseries"

    if isinstance(out, np.ndarray) and out.ndim == 1 and len(out) != len(index):
        return pd.DataFrame({f"{name}__array": out}), "profile"

    if isinstance(out, (list, tuple)):
        parts: list[pd.DataFrame] = []
        for position, item in enumerate(out):
            if isinstance(item, pd.Series):
                series = item.copy()
                series.name = f"{name}__{series.name or f'part{position}'}"
                parts.append(series.to_frame())
            elif isinstance(item, np.ndarray) and item.ndim == 1 and len(item) == len(index):
                parts.append(pd.DataFrame({f"{name}__part{position}": item}, index=index))
            else:
                raise TypeError(f"unsupported tuple/list item type: {type(item)!r}")

        if not parts:
            raise TypeError("empty tuple/list output")

        frame = pd.concat(parts, axis=1)
        kind = "timeseries" if len(frame) == len(index) else "profile"
        return frame, kind

    raise TypeError(f"unsupported output type: {type(out)!r}")


def _evaluate_interoperability() -> dict[str, Any]:
    scenario_names = [
        "trend_up",
        "trend_down",
        "sideways",
        "volatile",
        "gappy",
        "flat",
        "low_volume",
        "nan_streaks",
    ]
    scenarios = {
        scenario_name: _build_scenario(scenario_name, n=640, seed=100 + idx)
        for idx, scenario_name in enumerate(scenario_names)
    }

    callable_symbols: dict[str, Callable[..., Any]] = {}
    symbol_load_errors: dict[str, str] = {}

    for symbol in vi.__all__:
        if symbol == "__version__":
            continue
        try:
            obj = getattr(vi, symbol)
            if callable(obj):
                callable_symbols[symbol] = obj
        except Exception as exc:  # pragma: no cover
            symbol_load_errors[symbol] = f"{type(exc).__name__}: {exc}"

    results: dict[str, Any] = {}
    timeseries_outputs: dict[str, pd.DataFrame] = {}

    for symbol, func in callable_symbols.items():
        symbol_result = {"scenarios": {}}
        for scenario_name, data in scenarios.items():
            try:
                kwargs = _build_kwargs(func, data)
                out = func(**kwargs)
                out_df, out_kind = _normalize_output(symbol, out, data.index)

                if out_kind == "timeseries":
                    if len(out_df) != len(data):
                        raise ValueError(f"length mismatch: {len(out_df)} != {len(data)}")
                    if not out_df.index.equals(data.index):
                        raise ValueError("index mismatch")
                elif out_kind in {"profile", "levels"}:
                    if out_df.empty:
                        raise ValueError("empty non-timeseries output")
                else:
                    raise ValueError(f"unknown output kind: {out_kind}")

                symbol_result["scenarios"][scenario_name] = {
                    "ok": True,
                    "kind": out_kind,
                    "cols": int(out_df.shape[1]),
                }

                if scenario_name == "trend_up" and out_kind == "timeseries":
                    timeseries_outputs[symbol] = out_df
            except Exception as exc:
                symbol_result["scenarios"][scenario_name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        results[symbol] = symbol_result

    failed = {
        symbol: {
            scenario_name: entry["error"]
            for scenario_name, entry in symbol_result["scenarios"].items()
            if not entry["ok"]
        }
        for symbol, symbol_result in results.items()
        if not all(entry["ok"] for entry in symbol_result["scenarios"].values())
    }

    merged = pd.concat(timeseries_outputs.values(), axis=1) if timeseries_outputs else pd.DataFrame()

    return {
        "scenarios": scenarios,
        "results": results,
        "failed": failed,
        "symbol_load_errors": symbol_load_errors,
        "callable_symbols_tested": len(callable_symbols),
        "total_exported_symbols": len([s for s in vi.__all__ if s != "__version__"]),
        "merged_timeseries": merged,
    }


def _format_failures(failed: dict[str, dict[str, str]], max_symbols: int = 25) -> str:
    if not failed:
        return ""

    lines: list[str] = []
    for idx, symbol in enumerate(sorted(failed.keys())):
        if idx >= max_symbols:
            remaining = len(failed) - max_symbols
            lines.append(f"... and {remaining} more failing symbols")
            break

        scenario_errors = "; ".join(f"{name}: {error}" for name, error in sorted(failed[symbol].items()))
        lines.append(f"- {symbol}: {scenario_errors}")

    return "\n".join(lines)


@pytest.fixture(scope="module")
def interop_report() -> dict[str, Any]:
    return _evaluate_interoperability()


def test_cross_indicator_interop_matrix(interop_report: dict[str, Any]) -> None:
    report = interop_report

    assert not report["symbol_load_errors"], f"symbol load errors found: {report['symbol_load_errors']}"

    failures = report["failed"]
    assert not failures, "cross-indicator interoperability failures:\n" + _format_failures(failures)

    assert report["callable_symbols_tested"] == report["total_exported_symbols"]


def test_timeseries_outputs_merge_cleanly(interop_report: dict[str, Any]) -> None:
    report = interop_report
    merged = report["merged_timeseries"]
    trend_up_index = report["scenarios"]["trend_up"].index

    assert not merged.empty
    assert merged.index.equals(trend_up_index)
    assert int(merged.columns.duplicated().sum()) == 0


def test_cross_family_composition_stacks() -> None:
    data = _build_scenario("trend_up", n=640, seed=100)

    supertrend_df = vi.supertrend(data["high"], data["low"], data["close"])
    adx_df = vi.adx(data["high"], data["low"], data["close"])
    ema_series = vi.ema(data["close"], 50)
    rsi_series = vi.rsi(data["close"], 14)

    trend_follow_signal = (
        (data["close"] > ema_series)
        & (supertrend_df["SUPERTREND_DIR"] == 1)
        & (adx_df["ADX_14"] > 20)
        & (rsi_series > 50)
    )
    assert len(trend_follow_signal) == len(data)

    bbands_df = vi.bbands(data["close"], 20, 2.0)
    keltner_df = vi.keltner(data["high"], data["low"], data["close"], length=20, atr_length=10, mult=1.5)
    squeeze_signal = (bbands_df["BB_UPPER"] < keltner_df["KELTNER_UPPER"]) & (
        bbands_df["BB_LOWER"] > keltner_df["KELTNER_LOWER"]
    )
    assert len(squeeze_signal) == len(data)

    dss_df = vi.dual_score_signals(data["open"], data["high"], data["low"], data["close"], data["volume"])
    pc_df = vi.precision_confluence(data["open"], data["high"], data["low"], data["close"], data["volume"])
    rfc_df = vi.range_filter_confluence(data["high"], data["low"], data["close"])
    hrd_df = vi.htf_reversal_divergence(data["open"], data["high"], data["low"], data["close"])
    slp_df = vi.swing_leg_profile(data["open"], data["high"], data["low"], data["close"], data["volume"])

    new_stack_signal = (
        dss_df["DSS_BUY"]
        & pc_df["PC_BUY"]
        & rfc_df["RFC_LONG"]
        & (~hrd_df["HRD_BEAR_PATTERN"])
        & (slp_df["SLP_DIR"] == 1)
    )
    assert len(new_stack_signal) == len(data)

    ema_fast = vi.ema(data["close"], 12)
    ema_slow = vi.ema(data["close"], 26)
    short_run_series = vi.short_run(ema_fast, ema_slow)
    long_run_series = vi.long_run(ema_fast, ema_slow)
    decay_series = vi.decay(short_run_series.fillna(0.0), length=10)

    assert len(short_run_series) == len(data)
    assert len(long_run_series) == len(data)
    assert len(decay_series) == len(data)
