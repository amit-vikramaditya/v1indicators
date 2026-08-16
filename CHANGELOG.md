# Changelog

## 1.0.0 — Causality Release

The headline change of 1.0.0: **every per-bar indicator in this library is now
verified free of look-ahead bias by an automated prefix-invariance harness**
(`tests/test_causality.py`), and repainting pivot-family indicators default to
causal output.

### Breaking / behavioural changes

- **Pivot-family indicators are causal by default.** `support_resistance`,
  `equal_highs_lows`, `market_structure`, `support_resistance_breaks`,
  `support_resistance_channels`, `zigzag_swings`, `trendline_breaks`,
  `high_volume_levels`, `htf_reversal_divergence` now activate pivots, levels
  and signals at the bar where the pivot is *confirmed* (pivot bar plus the
  confirmation window). Pass `causal=False` for the legacy retrospective
  placement (plotting only — never for backtests).
- `lorentzian_knn` no longer suppresses its last `horizon` bars: outputs were
  previously dependent on where the series ends.
- `_step_resample`-based HTF patterns no longer emit from a trailing partial
  group.
- `hwc` outputs float dtype (was object dtype due to `pd.NA`).
- `ema`, `rma`, `sma`, `wma` gained `length=20` defaults (previously required).

### New indicators

- `parkinson`, `garman_klass` — classic range-based volatility estimators,
  optional trailing-window averaging.
- `fractional_difference` — Lopez de Prado memory-preserving stationarity
  transform (Numba kernel, weight truncation).
- `hurst` — rolling rescaled-range Hurst exponent (Numba kernel).
- `savgol` — one-sided (causal) Savitzky-Golay smoother: classic centred SG
  weights on a trailing window, i.e. the centred smooth delayed by
  (window-1)/2 bars.
- `variance_regime` — two-variance Gaussian likelihood-ratio LOW-volatility
  regime probability (explicitly not an HMM).
- `range_filter` — the ATR-scaled recursive hysteresis filter with trend
  state, extracted as a public API from `range_filter_confluence`
  (behaviour-preserving; golden-output verified).
- `candle_direction` — int8 +1/-1/0 OHLC primitive.

### Fixes and documentation

- **BREAKING:** `atr` now defaults to `mamode="rma"` — Wilder's original
  recursive smoothing, the textbook ATR (ta-lib / TradingView parity). The
  pre-1.0 behaviour is available with `mamode="ema"`. All ATR-based
  indicators (supertrend, range_filter, confluence engines, ...) follow the
  new default.
- Undocumented look-ahead bugs fixed in `support_resistance` and
  `equal_highs_lows` (pivots previously activated at the pivot bar, knowable
  only `right`/`length` bars later).
- **Removed** `dpo` (look-ahead by definition: price displaced backward by
  `length//2+1`; a visual detrending aid, never a causal signal) and `vp`
  (whole-series snapshot semantics, not a per-bar indicator).
- Reference-math parity tests pin exact textbook behaviour for EMA, RMA and
  CCI-with-MAD, and quantify the documented warmup-convention differences
  for RSI and ATR-rma.

### Removals

- `dpo`: look-ahead by definition (see above).
- `vp`: snapshot semantics incompatible with the per-bar causality contract.

### Quality gates

- `tests/test_causality.py` sweeps **every public function** with
  prefix-invariance checks across five cut points. The only permitted
  exception is documented in-code: `ichimoku` (Chikou span is look-ahead by
  textbook definition, strict xfail).

## 0.3.0

- Initial public release.
