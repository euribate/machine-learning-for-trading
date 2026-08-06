# `03_financial_features.ipynb` — cell-by-cell walkthrough

Companion notes for the ETF case study's feature-engineering stage. The
notebook itself explains *why* each section exists; this document explains
*what each cell does mechanically*, which functions it calls, and where
the values come from.

This stage turns the price panel into the 57-column matrix every model
downstream sees. Its central claim is not that the features predict — that
is `05_evaluation`'s question — but that every value was computable at the
moment it is dated.

---

## ✅ RESOLVED — the narrative digest no longer duplicates the artifact

**Status: fixed in the notebook and its paired script. No feature value
was touched.**

The summary markdown after section G stated a content digest of
`a1e90493a7de9d0f`, while the cell immediately below printed
`260b3945031e1849`. Every other figure in that sentence — 57 features,
396,186 rows, 99 ETFs, the date range, 22 clusters — agreed. Only the
digest had drifted.

**Why this mattered more than a typo.** Section G's whole rationale is
that a content digest moves when any feature value moves, precisely so a
corrected feature cannot pass silently downstream. A hand-typed digest in
the prose is a second copy of exactly that number, and it had already
drifted — the failure the digest exists to prevent, reproduced one cell
above the digest itself.

**The fix.** The clause now points at the printed value instead of
restating it:

```markdown
..., under the content digest the cell below prints. Cutting the
redundancy tree leaves **22 clusters**, ...
```

Correcting the hex string would have fixed this instance and left the
mechanism: a digest changes on every rebuild, so a copy of it in prose is
guaranteed to go stale again. It is also the one value in that sentence a
human reader cannot use — the counts and dates are meaningful on sight, a
sixteen-character hash is not, and it is printed immediately below.

Applied in both places, since the two are jupytext-paired and carried the
same sentence:

- `case_studies/etfs/03_financial_features.ipynb`, the `results`-tagged
  markdown cell
- `case_studies/etfs/03_financial_features.py:686`

The remaining counts and dates in that sentence are still hand-typed
copies of printed values. They are stable across reruns on fixed data, so
they have not drifted — but they are the same kind of duplication, and
`02_labels`'s audit record shows the alternative: build the statement from
the computed values so it cannot describe an artifact different from the
one written.

---

**Contents**

| # | Cell | What it does |
|---|---|---|
| 0 | Imports | Shared feature helpers and the `ml4t.engineer` indicators |
| 1 | Parameters | The single CI knob |
| 2 | Configuration | Windows, ranked list, thresholds from `setup.yaml` |
| 3 | Section A | The thesis, and the register that encodes it |
| 4 | Section B | Prices, eligibility, the yield curve |
| 5 | Section C.1 | Momentum, volatility, and their differences |
| 6 | Section C.2 | Oscillators, trend ratios, range |
| 7 | Section C.3 | Drawdown, volume, distance from extremes |
| 8 | Section C.4 | Cross-asset and macro state |
| 9 | The pipeline | Where the eligibility gate sits, and why |
| 10 | Section D.1–2 | The timing contract and the warmup audit |
| 11 | Section D.3 | Rebuilding without the holdout |
| 12 | Section E | Matrix assembly and the null policy |
| 13 | Figures F1, F4 | Coverage through time; the timing contract |
| 14 | Figures F2, F3 | Distributions; cross-sectional dispersion |
| 15 | Figure F5 | Redundancy clusters |
| 16 | Figure F6 | Persistence and rank stability |
| 17 | Section G | Emit, with digest sidecar |
| 18 | Takeaways | Four rules, four limitations |

---

## 0. Imports and helpers

- **Aim** — pull in the indicator implementations and the shared feature
  utilities, so this notebook writes only what is specific to ETFs.
- **Shows** — the split: `ml4t.engineer` owns the indicator conventions,
  `case_studies.utils.feature_engineering` owns the panel-safe statistics
  and the figures.
- **Outcome** — roughly thirty imported names, `CASE_DIR` and
  `FEATURES_DIR`.

```python
from ml4t.engineer.features.momentum import adx, aroon, cci, macd, rsi, stochastic
from case_studies.utils.feature_engineering import (
    EPS, assert_values_agree, assign_families, clip_within_date,
    cross_sectional_percentile, ..., warmup_audit,
)
```

The reason indicators are imported rather than written is given in C.2 and
is worth stating up front: **the smoothing convention inside an oscillator
is where two implementations of the same name diverge.** Wilder's
recursive average and a simple moving average of the same gains produce
different numbers, both correctly called "RSI". A shared implementation is
what makes `rsi_14` mean one thing across all nine case studies.

`EPS` recurs throughout as `.clip(lower_bound=EPS)` on every denominator.

## 1. Run parameters

- **Aim** — hold the one knob papermill overrides in CI.
- **Shows** — why there is no symbol cap here, unlike a stage that works
  per entity.
- **Outcome** — `START_DATE = None` in production.

```python
START_DATE = None
```

The notebook's comment carries the reasoning: the CI fixture is already
reduced in breadth, and **the cross-sectional families need a
cross-section to rank within**. Capping symbols would not merely shrink
the run, it would change the value of every percentile feature, because a
percentile is a property of the population it is taken over.

## 2. Configuration

- **Aim** — bind every window, threshold and list to `setup.yaml`.
- **Shows** — that the notebook invents no constant; a value typed here
  would be a second source of truth for a decision the rest of the
  pipeline reads from one place.
- **Outcome** — 10 declared families, decision cycle 21 sessions, holdout
  starting 2024-01-01.

```python
FAMILIES = families_from_config(setup)
WINDOWS = setup["features"]["windows"]
RANKED = setup["features"]["ranked"]
DECISION_CYCLE = int(resolve_label_horizon("etfs", setup["labels"]["primary"], setup).rstrip("Dd"))
```

`DECISION_CYCLE` comes through the same `resolve_label_horizon` chain
documented in the `02_labels` walkthrough, and it is used for one purpose:
fixing how far the persistence figure has to look. The logic is that **a
feature must hold its ordering for at least one decision cycle to be
usable at that cadence**, so the axis is set by the label, not chosen.

The concrete windows: momentum at `[5, 10, 21, 42, 63, 126, 189, 252]`,
volatility at `[21, 63, 126, 252]`, `skip_recent` 21, ranked columns
`["ret_126d", "sharpe_126d", "vol_63d"]`, regime threshold 0.005.

## 3. Section A — the thesis, and the register

- **Aim** — state the hypothesis precisely enough that the matrix's shape
  follows from it.
- **Shows** — three consequences: eight horizons rather than one, regime
  features that rank nothing, and percentile representations alongside raw
  levels.
- **Outcome** — the register table: one row per family with its inputs,
  lookback, lag and frame.

Three claims drive the design:

- **The carrier is trailing relative performance**, so returns and
  risk-adjusted returns are carried at *eight* horizons. Which horizon the
  effect lives at is treated as an empirical question rather than a
  modelling assumption. The skip-recent construction drops the most recent
  month from the long window, because that month reverses where the twelve
  before it continue.
- **The conditioning is regime.** Volatility, drawdown, trend strength,
  the equity-bond correlation and the curve shape rank nothing against
  each other; they say which environment a ranking is formed in. That
  distinction lives in the register's `role` column (`signal` vs `state`)
  and **no assertion can recover it from the values** — it is a statement
  about intent.
- **Representation matters as much as quantity.** A raw return's
  distribution drifts with the period's volatility, so the three carrier
  features are also carried as within-date percentiles.

**The lag column is the register's sharpest field.** A lag of 0 means the
input is on the tape at the decision itself, which every price-derived
family is: the decision is taken at the close and executes at the next
open. The yield curve is the sole exception at one session, and that lag
is *why it is a family of its own* rather than sharing one with the
cross-asset correlation, which reads prices.

## 4. Section B — inputs and their observability

- **Aim** — load the three inputs and note what each one's calendar
  implies.
- **Shows** — that the Treasury series and the price panel do not share a
  calendar, which dictates the join strategy in C.4.
- **Outcome** — 470,662 bars over 100 ETFs; 9,495 sessions of 10y–2y
  spread; the eligibility gate.

```python
yield_curve = (
    load_macro()
    .select("timestamp", ((pl.col("dgs10") - pl.col("dgs2")) / 100).alias("slope"))
    .drop_nulls().sort("timestamp")
)
```

Three observations the rest of the notebook depends on:

- **Bars are split- and dividend-adjusted**, so a trailing return spans a
  corporate action without a jump.
- **Eligibility is a per-year tradability gate**, stopping a fund listed
  in 2019 from appearing in a 2011 ranking. It arrives from
  `01_feasibility_analysis`.
- **The Treasury series skips market holidays.** This is why C.4 joins it
  backward in time rather than on an exact date — an equality join would
  drop every NYSE session that FRED did not publish on.

## 5. Section C.1 — momentum, volatility and their differences

- **Aim** — build the carrier block: trailing returns, volatility, and
  risk-adjusted returns at every declared horizon.
- **Shows** — what is shared with other case studies and what is specific
  to this one.
- **Outcome** — returns and volatility at eight and four horizons, plus
  skip-recent, acceleration and volatility-ratio columns.

```python
df = momentum_volatility_block(df, entity="symbol",
                              return_windows=WINDOWS["momentum"],
                              volatility_windows=WINDOWS["volatility"])
held = pl.col("close").shift(WINDOWS["skip_recent"]).over("symbol")
```

**Why the block is shared.** The notebook states that three case studies
had computed this "character for character with five different denominator
guards between them". Consolidating removes the divergence, not the
duplication.

**Skip-recent is the interesting construction.** `held` is the close 21
sessions ago, and `skip_recent_12_1` divides it by the close 252 sessions
ago. So the window runs from t−252 to t−21 and deliberately **excludes the
most recent month**, isolating the twelve-month continuation from the
one-month reversal that would otherwise contaminate it.

The acceleration columns (`mom_accel_short/medium/long`) are differences
between adjacent horizons, and the `vol_ratio_*` columns are short
volatility over long. Both are cheap ways to express *change* in the
carrier without adding a new lookback.

## 6. Section C.2 — oscillators, trend ratios and range

- **Aim** — add bounded oscillators and moving-average ratios from the
  shared indicator library.
- **Shows** — that indicator identity depends on smoothing convention, so
  the implementation is imported rather than rewritten.
- **Outcome** — RSI at 7 and 14, MACD, ADX, CCI, stochastic, Aroon, NATR,
  choppiness, Hurst, SMA ratios at 50/200, an EMA ratio, Bollinger %B.

Two details in this cell are load-bearing.

**The Hurst exponent is the only rounded column**, and the notebook
explains why at length. It is the slope of a log-log least-squares fit
over rescaled-range statistics, so its last bits depend on the order the
cumulative sums accumulate in — which differs between a full panel and a
truncated one on some BLAS builds. On this data the two agree exactly, but
CI trips D.3's `1e-12` tolerance. Rounding to six decimals is "far below
any reading of a persistence exponent and far above the noise", so **D.3
tests the feature rather than the platform**. This is a deliberate,
documented weakening of one assertion, not an oversight.

**Bollinger %B is guarded by a `when`**, not a clip:

```python
pl.when(pl.col("_sd") > 0)
  .then((pl.col("close") - (pl.col("_mid") - 2 * pl.col("_sd"))) / (4 * pl.col("_sd")))
```

With no `otherwise`, a zero-dispersion window yields `null` rather than a
fabricated number. That is the same choice `02_labels` makes for an
incomplete forward window, for the same reason.

## 7. Section C.3 — drawdown, volume and distance from extremes

- **Aim** — add state features describing where price sits relative to its
  own recent range.
- **Shows** — that `max_dd_63d` is the *current* drawdown, not the worst
  decline in the window; these are different statistics.
- **Outcome** — drawdown at 63 and 126, OBV z-score, positive-day share,
  and distance from the 52-week high and low.

`max_dd_63d` is the share by which price sits below its highest close of
the trailing quarter — **zero at a new high and negative otherwise**. The
maximum drawdown *within* the window is a different quantity, and the
notebook names the distinction explicitly so a reader does not assume the
more familiar one.

Relative volume is computed here but **clipped later**, at the 1st and
99th percentile of its own date. The reason is that an index rebalance
puts one ETF orders of magnitude above its own average, and one such row
otherwise sets the scale every model sees.

## 8. Section C.4 — cross-asset and macro state

- **Aim** — add the two features that are one number per date, shared by
  every ETF.
- **Shows** — two calendar hazards: a rolling window that reads row order,
  and two series with different holiday calendars.
- **Outcome** — `corr_spy_tlt_63d`, `regime`, `yield_curve_slope`,
  `yield_curve_zscore`, broadcast to every row.

**The re-sort is not cosmetic.** `pl.rolling_corr` reads *row order*, not
the timestamp column. The SPY/TLT pair is assembled by an inner join,
which does not guarantee order, so `.sort("timestamp")` runs before the
rolling correlation. Without it the window would silently correlate
whatever rows happened to be adjacent.

**The curve is stamped with its availability date, not its observation
date:**

```python
curve = yield_curve.select(
    pl.col("timestamp").dt.offset_by("1d"), ...
)
...
.join_asof(curve, on="timestamp", strategy="backward")
```

Two decisions are packed in there, and the notebook is explicit about
both:

- **`offset_by("1d")` is a calendar day, not a shift down the series.** A
  shift would mean "the next day the Treasury published", so on Columbus
  Day — when NYSE trades and FRED does not — it would add a *second*
  session of delay. The policy `setup.yaml` declares is
  `alfred_initial_release_close_lagged`: a value dated `t` is available at
  the close of `t+1`.
- **`join_asof(..., strategy="backward")`** carries the most recent value
  at or before the row's timestamp, so a market holiday inherits the
  previous session's spread and never a later one's.

## 9. The pipeline — where the eligibility gate sits

- **Aim** — compose the family functions into one build, in an order that
  is itself a correctness claim.
- **Shows** — that the gate must sit *after* per-entity work and *before*
  anything computed within a date.
- **Outcome** — `build_features`, and a built panel of 396,186 eligible
  bars carrying 57 features.

```python
def build_features(df):
    return df.pipe(per_entity_features).pipe(gate_to_eligible).pipe(clip_and_rank)
```

**This ordering is the most consequential decision in the notebook**, and
both halves of it are argued:

- **The gate goes before `clip_and_rank`** because a percentile and a clip
  are properties of the cross-section they are taken over. An ETF the
  strategy cannot trade must not be in that cross-section — ranking
  against it *moves the number written for every ETF that is*.
- **The gate goes after `per_entity_features`**, and the relative-volume
  ratio is specifically placed on the earlier side. Its trailing mean has
  to read every bar the ETF traded; otherwise a fund admitted this year
  divides by a mean of its first few eligible days, and one that re-enters
  after a gap averages across the gap.

So the rule is: **anything reading one entity's own history runs on the
full panel; anything reading across entities runs on the gated panel.**
Only the clip is a cross-sectional step in that block, which is why it is
separated from the ratio that feeds it.

`EXCLUDED` drops `symbol`, `timestamp`, OHLC, `volume` and `log_return`
from the feature list — the inputs, not the features.

## 10. Section D.1–D.2 — the timing contract and warmup

- **Aim** — classify every operation used, then assert that trailing
  windows do not produce values before they could have filled.
- **Shows** — three operation kinds (rolling, cross-sectional, as-of), and
  a per-column check that leading nulls match declared lookbacks.
- **Outcome** — a nine-row audit, every column `populated = true`.

```python
warmup_audit(per_entity, {"ret_252d": 252, "sma_ratio_200": 200, ...}, entity="symbol")
```

**The audit asserts rather than describes**: a column carrying a value
before its window could have filled is reading bars that do not exist.

**It runs on `per_entity`, before the gate** — deliberately, because the
gate drops exactly the early rows the warmup stretch is made of. Auditing
the gated panel would test nothing.

Reading the output, `ret_252d` first populates at bar **253** while
`dist_52w_high` populates at **252**, both declared 252. That is not an
inconsistency: a 252-session *return* needs 253 bars because it compares
two endpoints, whereas a rolling *max* over 252 bars is complete at the
252nd. The audit passes both because it checks the declared lookback is
not *undershot*.

## 11. Section D.3 — rebuilding without the holdout

- **Aim** — prove no transform in the whole construction fits across the
  sample.
- **Shows** — that building the panel twice, once truncated, reproduces
  identical values on shared rows.
- **Outcome** — 349,500 rows compared per column, `0` nulls on one side
  only, max abs difference `0.0`.

```python
seal = assert_values_agree(
    built.filter(pl.col("timestamp") < HOLDOUT_START),
    build_features(prices.filter(pl.col("timestamp") < HOLDOUT_START)),
    columns=feature_cols, keys=["timestamp", "symbol"],
)
```

**This is the strongest check in the notebook**, and its strength comes
from what it does *not* require. A trailing statistic is unaffected by
truncation; a fitted parameter — a winsorization bound, a scaler, an
encoder — is not, because truncating the column moves the parameter and
with it every row. So rebuilding and comparing tests **every emitted
column at once, and does not depend on anyone having remembered to flag
the transform that fits**.

Two design choices make the test honest:

- **A value against a null counts as a difference.** A null-skipping
  comparison would hide exactly the failure where truncation changes
  whether a value exists.
- **All 57 columns are compared, not a sample.** The frame shown in the
  notebook filters to three representative columns for display only.

Note this seal covers the *construction*. It says the features are
trailing; it says nothing about whether they predict.

## 12. Section E — matrix assembly and the null policy

- **Aim** — select the final columns, apply one null policy, and assert
  the panel key is unique.
- **Shows** — that raw inputs are excluded so a model cannot read its own
  answer.
- **Outcome** — the `features` frame, and the register re-rendered with
  realized column counts per family.

```python
features = (
    built.select(["timestamp", "symbol", *feature_cols])
    .drop_nulls(subset=["sharpe_126d"])
    .sort(["timestamp", "symbol"])
)
assert features.select(["timestamp", "symbol"]).is_duplicated().sum() == 0, "duplicate panel key"
```

**Excluding `log_return` matters more than it looks.** A model handed the
contemporaneous log return beside a label derived from the same prices
would be reading its own answer.

**The null policy is currently a no-op, and that is worth knowing.** The
gated panel has 396,186 rows and `features` has 396,186 rows, so
`drop_nulls(subset=["sharpe_126d"])` removes nothing. The reason is that
the eligibility gate already requires a prior full year above the volume
floor, so a fund's first eligible bar is at least a year after listing —
well past a 126-session warmup. The line is not dead: it would bite if the
gate were relaxed or the ranked list moved to a longer window. But as
configured, the gate subsumes it, and the "one null policy applied once"
the markdown describes is doing no filtering today.

## 13. Figures F1 and F4 — coverage and the timing contract

- **Aim** — show that no family is materially incomplete, and draw each
  family's lookback and lag.
- **Shows** — where residual thinness sits, and which family waits for its
  input to publish.
- **Outcome** — no family below about 99% coverage; one bar short of the
  decision line.

**F1's axis is scaled to the data, not pinned to zero**, and the notebook
justifies it: on a zero-based axis every family would draw as one flat
line at the top, and the point is *where the residual percent sits*. It
sits in the long-window families, because an ETF admitted partway through
a year has not yet filled a 252-session lookback.

**F4 draws the register, not the code.** Every family's bar reaches the
decision line except the yield curve, which stops one session short — the
lag from section 3 made visible. A gap at the right edge is an information
lag by construction.

## 14. Figures F2 and F3 — distributions and dispersion

- **Aim** — read the scale each feature arrives on, and whether the
  cross-section disagrees enough to rank.
- **Shows** — that a longer window widens the return and narrows the
  ratio; and that dispersion widens sharply in stress.
- **Outcome** — six histograms; a 10th–90th percentile band through time.

The F2 pairing is the informative part: trailing returns broaden with
horizon (about ±0.2 at 21 sessions to a body reaching 0.75 at 252) while
the annualized risk-adjusted returns move the *other way*, narrowing from
roughly −5…10 to −2…3. Dividing by dispersion is what stabilises the
scale.

F3's premise is stated in one line worth keeping: **on a date where the
band narrows to nothing there is nothing to rank**, whatever the average
level of the feature.

## 15. Figure F5 — redundancy structure

- **Aim** — find how much of the matrix is one ordering under several
  names.
- **Shows** — clustering on `1 - |ρ|`, so features carrying the same
  ordering group regardless of sign.
- **Outcome** — 22 clusters from 57 features.

Above the 0.7 cut, two features are close enough that a linear model
cannot separate their contributions. Adjacent horizons pair off tightly —
the 5- and 10-day returns with their risk-adjusted twins — while the
short-horizon and long-horizon blocks only merge near the root.

The notebook is careful about scope: this **states** the clusters.
Choosing one representative from each needs a fold-aware criterion and
belongs to `05_evaluation`. Picking representatives here, on the full
sample, would be a selection fitted across the data the seal in section 11
just proved nothing else fits across.

## 16. Figure F6 — persistence and rank stability

- **Aim** — measure how long a feature's value and its ordering last.
- **Shows** — autocorrelation to two decision cycles, and rank correlation
  between consecutive rebalances.
- **Outcome** — long-window carriers still above 0.7 at 42 sessions; the
  21-day return reaches zero at exactly 21.

```python
DECISION_DATES = (
    features.group_by(pl.col("timestamp").dt.truncate("1mo"))
    .agg(pl.col("timestamp").max().alias("decision"))["decision"].sort().to_list()
)
```

**Rebalances, not a fixed lag.** `setup.yaml` declares
`monthly_month_end`, and consecutive month-ends are a *varying* number of
sessions apart. A fixed lag would correlate dates the strategy never puts
side by side.

**The autocorrelation is per ETF, then summarised by the median**, with a
bootstrap interval over ETFs. Pooling over every ETF-date pair would read
high whenever ETFs sit at different levels, whether or not any one of them
persists — the same panel hazard `01_feasibility_analysis` avoids with
`panel_acf`.

The most legible result: **`ret_21d` reaches zero autocorrelation at
exactly 21 sessions, the length of its own window**, and its
rebalance-to-rebalance rank correlation is almost nothing. A feature whose
ordering decays inside a cycle cannot support that rebalance cadence
however well it predicts on the day it is computed.

## 17. Section G — emit

- **Aim** — write the matrix with enough provenance to distinguish two
  vintages.
- **Shows** — a content digest over values, not file bytes.
- **Outcome** — `features/financial.parquet` plus a `.digest.json`
  sidecar recording three input digests.

```python
record = write_artifact(
    features, FEATURES_DIR / "financial.parquet",
    keys=["symbol", "timestamp"],
    written_by="case_studies/etfs/03_financial_features.py",
    inputs={"eligibility.csv": ..., "load_etfs": ..., "load_macro:dgs10-dgs2": ...},
)
```

The digest is computed over content, so row order and parquet metadata
leave it alone while **any feature value moves it**. The notebook names
the gap this fills: a feature-set *name* reaches the registry, a
feature-set *value* does not, so a corrected feature would otherwise move
every number downstream without changing anything the registry stores.

The prose summarising this section used to restate the digest by hand and
had gone stale; it now points at the printed value — see the resolved
issue at the top.

## 18. Key takeaways and limitations

- **Aim** — state the four transferable rules and the four places the
  matrix is weaker than it looks.
- **Shows** — each rule as a failure avoided.
- **Outcome** — the handoff to `04_model_based_features` and
  `05_evaluation`.

The rules: state the timing contract before writing the feature; test the
seal by construction rather than inspection; rank inside the date; read
the matrix for distribution, dispersion, redundancy and decay before
modelling it.

The limitations, all stated by the notebook:

- **The cross-asset regime feature is one pair**, SPY against TLT. It
  describes the equity-bond relationship and says nothing about
  commodities or currencies — which are a third of the universe.
- **The eligibility gate is annual**, so an ETF that lost liquidity in
  June stays in the cross-section until December.
- **The yield curve reads revised history, not the initial release.** The
  configured one-session lag is applied honestly, but a value revised
  later is not the value the decision could have seen, whatever its
  timestamp says. This is the subtlest of the four: the timing contract is
  satisfied in form while the underlying number may not be the
  contemporaneous one.
- **Every feature here is a rule written in advance.**
  `04_model_based_features` adds features that are themselves model
  outputs, where the rule is estimated from the data.

**Next**: `04_model_based_features` builds regime and memory-preserving
features on top of this matrix; `05_evaluation` tests fold by fold whether
any of it predicts.
