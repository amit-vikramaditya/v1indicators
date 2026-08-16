# Changelog

## 1.0.0 — Causality Release (unreleased on PyPI; latest published: 0.3.0)

The headline change of 1.0.0: **every per-bar indicator in this library is now
verified free of look-ahead bias by an automated prefix-invariance harness**
(`tests/test_causality.py`), and repainting pivot-family indicators default to
causal output. The harness sweeps **every public function** across multiple
synthetic seeds and non-default parameter values; there are **no permitted
look-ahead exceptions**.

### Breaking / behavioural changes

- **Pivot-family indicators are causal by default.** `support_resistance`,
  `equal_highs_lows`, `market_structure`, `support_resistance_breaks`,
  `support_resistance_channels`, `zigzag_swings`, `trendline_breaks`,
  `high_volume_levels`, `htf_reversal_divergence` now activate pivots, levels
  and signals at the bar where the pivot is *confirmed* (pivot bar plus the
  confirmation window). Pass `causal=False` for the legacy retrospective
  placement (plotting only — never for backtests).
- **`ichimoku` is causal by default.** The Chikou column is emitted as the
  spread it encodes (`close - close.shift(kijun)`), which carries the identical
  signal information with no future data. `causal=False` restores the textbook
  displaced construction (plotting only). The look-ahead exception registry
  (`KNOWN_LOOKAHEAD`) is now empty.
- `_step_resample`-based HTF patterns no longer emit from a trailing partial
  group.
- `hwc` outputs float dtype (was object dtype due to `pd.NA`).
- `ema`, `rma`, `sma`, `wma` gained `length=20` defaults (previously required).
- `lorentzian_knn` no longer suppresses its last `horizon` bars.

### Removals (duplicates and look-ahead-by-definition)

- `dpo`: look-ahead by definition (price displaced backward by `length//2+1`;
  a visual detrending aid, never a causal signal).
- `vp`: whole-series snapshot semantics, incompatible with the per-bar
  causality contract.
- Exact duplicate indicators, one canonical name kept for each:
  `adl` (identical to `ad`), `zlma` (identical to `zlema`), `tma` (identical
  to `trima`), `vpt` (identical to `pvt`), `median` (identical to
  `quantile(q=0.5)`). The `ma` dispatcher still accepts `mamode="zlma"`,
  now mapped to `zlema`.

### Calendar and session fixes

- **`day_week_month_levels`: prior-day levels are no longer NaN on Mondays**
  (and after full-day holidays). Empty calendar bins (weekends/holidays) are
  skipped before the shift, so "prior day" is the prior *trading* day.
- **`day_week_month_levels`: monthly levels fixed off-by-one** — every bar saw
  the month-before-last's aggregate (month-end resample labels); bars now see
  the last *completed* month.
- **`session_range`: overnight sessions (`start > end`) now work** — previously
  produced a silent empty mask (all-NaN output). Post-midnight bars join the
  session that started the previous calendar day, so running bounds accumulate
  across midnight.
- **`session_killzones`: wrapped (cross-midnight) zones group correctly** —
  post-midnight bars previously started a fresh cummax group at 00:00.

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
  pre-1.0 behaviour is available with `mamode="ema"`. Indicators that call
  `atr` without an explicit mode follow the new default (`supertrend`,
  `ut_bot`, ...); `range_filter` and `natr` deliberately pin `"ema"` to
  preserve their pre-1.0 outputs (documented in their docstrings).
- Reference-math parity tests pin exact textbook behaviour for EMA, RMA and
  CCI-with-MAD, and quantify the documented warmup-convention differences
  for RSI and ATR-rma.
- Documented conventions: warmup (exponential family emits from bar 0),
  boolean flags are `False` on NaN inputs, `vwap` is anchored to the input's
  first bar (no automatic session reset), `pivot_points` recomputes from the
  prior *bar* (feed daily OHLC for classic daily pivots).
- Requires Python >= 3.10 (the previous `>=3.9` claim was incorrect —
  PEP 604 annotations crash on 3.9 import).

### Quality gates

- `tests/test_causality.py` sweeps **every public function** with
  prefix-invariance checks across five cut points and multiple seeds.
- `tests/test_causality_params.py` re-runs the causality property under
  non-default parameter values (window extremes, alternate modes, step sizes).
- `tests/test_interoperability_matrix.py` exercises every symbol on eight
  market scenarios plus a weekday-only calendar scenario with weekend gaps.
- `tests/test_calendar_sessions.py` pins the calendar/session behaviours
  above on real-market data shapes (weekday-only indices).
- GitHub Actions test workflow (`.github/workflows/tests.yml`) runs the full
  suite on Python 3.10 / 3.12 / 3.14.

## 0.3.0

- Initial public release.
