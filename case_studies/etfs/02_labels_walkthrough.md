# `02_labels.ipynb` — cell-by-cell walkthrough

Companion notes for the ETF case study's label-engineering stage. The
notebook itself explains *why* each section exists; this document
explains *what each cell does mechanically*, which functions it calls,
and where the values come from.

The configuration cell (section 0 below) is documented in depth — it is
the cell that binds the notebook to `config/setup.yaml`, and the
resolution chain behind it is not obvious from reading the line.
Everything else is covered at normal depth.

---

## ⚠️ OPEN ISSUE — section G will raise `TypeError` on re-run

**Status: unresolved, awaiting review. Nothing has been changed in the
notebook or the environment.**

The section G cell calls:

```python
stats = compute_ic_hac_stats(
    ic, ic_col="ic", label_horizon=PRIMARY_HORIZON
)
```

The `ml4t_diagnostic` build installed in `.venv` (**0.1.0b21**) exposes:

```python
compute_ic_hac_stats(ic_series, ic_col="ic", maxlags=None,
                     kernel="bartlett", use_correction=True)
```

There is **no `label_horizon` parameter**, and the signature takes no
`**kwargs`, so the call fails outright:

```
TypeError: compute_ic_hac_stats() got an unexpected
           keyword argument 'label_horizon'
```

Verified by introspection against the active environment:

```bash
.venv/bin/python -c "
import inspect
from ml4t.diagnostic.metrics import compute_ic_hac_stats as f
print(inspect.signature(f))
"
```

**What this means.** The stored outputs in the notebook (mean IC 0.0203,
naive t 3.77, HAC t 1.08, p 0.282) were produced against a **newer**
`ml4t_diagnostic` than the one now installed. The notebook currently
executes cleanly up to section G and then stops there.

**Two possible fixes — not applied, decision pending:**

1. **Upgrade `ml4t_diagnostic`** to the build that accepts
   `label_horizon`. Correct if that parameter does something the
   notebook depends on — most likely setting the HAC lag window from the
   label horizon (21) rather than from the Newey–West rule `floor(4 *
   (T/100)^(2/9))`. This preserves the recorded numbers.
2. **Change the call to `maxlags=PRIMARY_HORIZON`** against the
   installed version. Only equivalent if `label_horizon` was in fact
   just a lag-setter; if the newer build derives the lag differently
   (e.g. `horizon - 1`, or a multiple), the HAC t-statistic will shift
   and the markdown commentary quoting 1.08 will no longer match the
   output.

Option 1 is the safer default, since option 2 risks silently changing
the reported significance of the baseline floor — the number the whole
section exists to establish. Confirming which requires reading the newer
build's source; I have not done so.

---

**Contents**

| # | Cell | What it does |
|---|---|---|
| 0 | Configuration | Resolves label names and horizons |
| 1 | Imports | Modules, `CASE_DIR`, `LABELS_DIR` |
| 2 | Parameters | Run knobs for CI vs production |
| 3 | Section B | Load prices, digest the input |
| 4 | Section C | Build the forward-return labels |
| 5 | Section D | Four assertions on window validity |
| 6 | Figure F2 | Boundary profile — the null tail |
| 7 | Section E | Seal on the label endpoint |
| 8 | Figure F1 | Root-horizon scaling check |
| 9 | Figure F4 | Cross-sectional dispersion by year |
| 10 | Figure F3 | Overlap decay, effective sample size |
| 11 | Section G | Baseline IC floor — ⚠️ **open issue** |
| 12 | Section H | Write parquet + digest sidecars |
| 13 | Audit record | Printed label definition table |

---

## 0. The configuration cell (in depth)

```python
setup = yaml.safe_load((CASE_DIR / "config" / "setup.yaml").read_text())

PRIMARY_LABEL = setup["labels"]["primary"]
LABEL_NAMES = [PRIMARY_LABEL, *setup["labels"].get("variants", [])]
HORIZONS = {
    n: int(resolve_label_horizon("etfs", n, setup).rstrip("Dd"))
    for n in LABEL_NAMES
}
HOLDOUT_START = date.fromisoformat(setup["evaluation"]["holdout_start"])
PRIMARY_HORIZON = HORIZONS[PRIMARY_LABEL]
VARIANT_LABEL = LABEL_NAMES[1]
```

### What comes out

```python
PRIMARY_LABEL   = "fwd_ret_21d"
LABEL_NAMES     = ["fwd_ret_21d", "fwd_ret_5d"]
HORIZONS        = {"fwd_ret_21d": 21, "fwd_ret_5d": 5}
PRIMARY_HORIZON = 21
VARIANT_LABEL   = "fwd_ret_5d"
```

Two names and two numbers. Everything below in the notebook is a loop
over that dict.

### Why the `HORIZONS` line is written this way

The notebook needs the integer `21` to build the label:

```python
pl.col("close").shift(-horizon)   # look 21 sessions ahead
```

The only real question is *where `21` comes from*:

```python
# Option A — type it into the notebook
HORIZONS = {"fwd_ret_21d": 21, "fwd_ret_5d": 5}

# Option B — what the notebook does
HORIZONS = {
    n: int(resolve_label_horizon("etfs", n, setup).rstrip("Dd"))
    for n in LABEL_NAMES
}
```

Option A creates a **second copy of the number**. The rest of the
pipeline — cross-validation windows (`case_studies/utils/cv_window.py`),
the backtest, stage 03's feature evaluation — reads the horizon from
`setup.yaml`. Change the strategy to a two-month hold, edit
`setup.yaml`, and Option A would keep labelling at 21 while the CV purge
gap moved to 42. No exception, no warning, just a leak. Option B keeps
**one source of truth**.

### The resolution chain, step by step

`resolve_label_horizon` (`utils/artifact_specs.py:125`) tries three
sources in order and returns the first hit:

```python
def resolve_label_horizon(case_study_id, label, setup=None):
    # source 1
    label_spec = load_label_spec(case_study_id, label)
    if label_spec is not None:
        definition = label_spec.get("definition", {})
        if definition.get("horizon"):
            return str(definition["horizon"])

    # source 2
    labels = (setup or {}).get("labels", {})
    horizon = (labels.get("horizons") or {}).get(label)

    # source 3
    if horizon:
        return str(horizon)
    return resolve_label_buffer(case_study_id, label, setup)
```

**Source 1 — a per-label spec file.** Would live at
`case_studies/etfs/config/artifacts/labels/fwd_ret_21d.yaml`, field
`definition.horizon`. The `etfs` config directory contains only
`backtest/`, `exploration/`, `training/` and `setup.yaml` — there is no
`artifacts/` directory. `load_label_spec` returns `None`; branch
skipped.

**Source 2 — an explicit `horizons:` block in `setup.yaml`.** For this
to fire the file would need:

```yaml
labels:
  primary: fwd_ret_21d
  # ← this block does not exist in etfs/config/setup.yaml
  horizons:
    fwd_ret_21d: 21D
    fwd_ret_5d: 5D
```

There is no `horizons:` key, so `labels.get("horizons")` is `None`,
`(None or {}).get(label)` is `None`, and the `if horizon` test fails.

**Source 3 — the fallback, `resolve_label_buffer`**
(`utils/artifact_specs.py:103`). This is what actually runs. It reads
the **CV buffer** fields:

```yaml
labels:
  primary: fwd_ret_21d
  buffer: 21D                    # ← used for the primary label
  variants:
    - fwd_ret_5d
  variant_buffers:
    fwd_ret_5d: 5D               # ← used for the variant
```

So both horizons in this case study come from buffer fields, not horizon
fields.

### Horizon vs buffer — two different things

- **horizon** — how far forward the label looks:
  `close[t+21] / close[t] - 1`
- **buffer** — the gap dropped between train and test folds, so the last
  training label has finished resolving before the test window starts

They are usually equal and here they are — both 21. They need not be.
You might use a 5-day horizon but purge 21 days between folds because a
*feature* has a 21-day lookback that would leak. Different jobs,
separate config fields.

The notebook's markdown says exactly this:

> They are separate fields — the buffer that keeps folds
> independent is not always the horizon the outcome resolves
> over — and coincide here.

**The risk it is flagging:** because source 2 is absent, the horizon is
being read out of the buffer field. If someone later sets `buffer: 30D`
as a safety margin while still intending a 21-day label, this cell would
silently start building **30-day labels**. Adding an explicit
`horizons:` block (source 2) closes that hole.

### `.rstrip("Dd")` and `int(...)`

`setup.yaml` stores durations as strings with a unit suffix (`21D`,
`5D`) because that is the format the backtest and CV code also consume.
Converting to something `shift()` accepts:

```
"21D"  →  rstrip("Dd")  →  "21"  →  int()  →  21
```

`rstrip` takes a **set of characters**, not a suffix — it strips any of
them from the right end repeatedly:

```python
"21D".rstrip("Dd")   → "21"
"5d".rstrip("Dd")    → "5"     # lowercase handled too
"21DD".rstrip("Dd")  → "21"    # strips both
```

Practically, `setup.yaml` may write `21D` or `21d`. The downside of a
char set is that it strips blindly: a value like `21W` (weeks) or `3M`
(months) survives `rstrip` intact and then explodes on `int()`. It only
handles the day convention.

### What `variants:` controls

`LABEL_NAMES` is built from `setup["labels"]["variants"]`, so that YAML
list decides **how many labels the notebook builds**. Add `fwd_ret_63d`
to `variants` plus a matching `variant_buffers: {fwd_ret_63d: 63D}`
entry, and without touching notebook code you get a third parquet file,
a third row in every loop, a third curve on the boundary-profile chart.

### Failure mode worth knowing

`resolve_label_horizon` is typed `-> str | None`, and all three sources
can miss. Add a name to `variants` **without** a matching
`variant_buffers` entry (and without an explicit `horizons` entry) and
this line dies with:

```
AttributeError: 'NoneType' object has no attribute 'rstrip'
```

— an unhelpful message for what is really a missing-config error.

### The remaining assignments

- `HOLDOUT_START =
  date.fromisoformat(setup["evaluation"]["holdout_start"])` — parses the
  ISO date string into a `datetime.date` so it can be compared against
  Polars date columns in section E.
- `PRIMARY_HORIZON` / `VARIANT_LABEL` — convenience handles, so figure
  code reads `PRIMARY_HORIZON` instead of
  `HORIZONS[setup["labels"]["primary"]]`.
- `VARIANT_LABEL = LABEL_NAMES[1]` assumes at least one variant. With
  `variants:` emptied, this raises `IndexError` — and figure F1, which
  compares exactly two labels, would need rewriting anyway.

---

## 1. Imports and paths

```python
CASE_DIR = get_case_study_dir("etfs")
LABELS_DIR = CASE_DIR / "labels"
```

`get_case_study_dir` (`utils/paths.py`) resolves the case-study root
from the repo layout rather than from the notebook's own location, so
the notebook works the same whether run in Jupyter or through papermill
via `02_labels.py`.

The imports fall into four groups:

- **`ml4t.diagnostic.metrics`** — `cross_sectional_ic_series`,
  `compute_ic_hac_stats` (installed package, used in section G).
- **`case_studies.utils`** — `value_digest` / `write_artifact` (content
  hashing and the sidecar format), `effective_sample_size` /
  `panel_autocorrelation` (overlap diagnostics).
- **`data.load_etfs`** — the adjusted daily bars, verified in
  `01_feasibility_analysis`.
- **`utils`** — `resolve_label_horizon`, `get_case_study_dir`, and the
  shared plot style (`COLORS`, `FIGSIZE`, `add_message_title`).

`warnings.filterwarnings("ignore")` keeps the rendered notebook clean.

## 2. Run parameters

```python
MAX_SYMBOLS = None
START_DATE = None
MIN_SYMBOLS_FOR_DISPERSION = 10
```

Papermill-injectable knobs. Production leaves the first two as `None`;
CI overrides only `START_DATE` to shorten the run. `MAX_SYMBOLS`
deliberately stays unset even in CI — stage 03 needs 10+ non-null labels
per date to compute a cross-sectional IC, and thinning the universe
would break that.

`MIN_SYMBOLS_FOR_DISPERSION` is used only by figure F4, which takes a
standard deviation across symbols per date. Across two or three symbols
that number is noise, so thin dates are dropped.

## 3. Section B — load and digest the prices

```python
prices = (
    load_etfs()
    .select(["symbol", "timestamp", "close"])
    .sort(["symbol", "timestamp"])
)
```

The sort is load-bearing: `shift(-h).over("symbol")` means "h rows down
within this symbol", which is only "h sessions forward" if the rows are
in date order within symbol.

**No eligibility filter is applied here, and the ordering is the
point.** Dropping ineligible rows first would make `shift` count
*surviving* rows, so the horizon would stop being measured in trading
sessions and the label would silently span the gap. Eligibility
(`eligibility.csv`) is applied later — to the section G baseline, and to
the trainable panel in `03_financial_features`.

```python
MARKET_DATA_DIGEST = value_digest(
    prices, ["symbol", "timestamp", "close"]
)
```

`value_digest` (`case_studies/utils/artifact_digest.py`) hashes each
row, sorts the row hashes, then hashes the result — so the digest is
invariant to row order and to parquet metadata churn, but sensitive to
any value change. It is recorded as every label's `inputs` in section H:
without it, a re-run against a refreshed download is indistinguishable
from this one.

## 4. Section C — build the labels

```python
def forward_return(df, horizon, name):
    return df.with_columns(
        (
            pl.col("close").shift(-horizon).over("symbol")
            / pl.col("close")
            - 1
        ).alias(name)
    )
```

Close-to-close forward return, `P[t+h]/P[t] - 1`, Chapter 7.2's
convention. `.over("symbol")` confines the shift inside each ETF, so no
window crosses an entity boundary — asserted in section D, property 3.
Rows without a full forward window get `null` from the shift, which is
the correct representation of "unknown", not zero.

### The two bookkeeping columns

Two bookkeeping columns are numbered here, on the **complete** price
series, because both mean something only before any row is dropped:

```python
labels_df = prices.with_columns(
    (pl.len().over("symbol") - 1 - pl.int_range(pl.len()).over("symbol")).alias("from_end"),
    pl.int_range(pl.len()).over("symbol").alias("session"),
)
```

**`.over("symbol")`** is a window partition — SQL's
`OVER (PARTITION BY symbol)`. Polars evaluates the expression once per
symbol and scatters the results back to the rows they came from, so the
frame keeps its shape and its order. Nothing is grouped away.

Read the two pieces separately:

- **`pl.int_range(pl.len())`** — inside `.over()`, `pl.len()` is *that
  symbol's* row count, so the range yields `0, 1, 2, … n-1` down the
  group. That is `session`: each symbol's bars numbered forward from its
  own first bar, independently of every other symbol.
- **`pl.len().over("symbol")`** — the same group count, but broadcast
  unchanged onto every row of the group. Subtracting the session number
  from `n - 1` flips the count around: `from_end` is `0` on the symbol's
  last bar, `1` on the one before it, and so on backwards.

With `horizon = 2` and two symbols of 5 and 3 bars:

| symbol | close | session | from_end | forward return |
|--------|-------|---------|----------|----------------|
| AAA | 10 | 0 | 4 | 0.200 |
| AAA | 11 | 1 | 3 | 0.182 |
| AAA | 12 | 2 | 2 | 0.167 |
| AAA | 13 | 3 | 1 | `null` |
| AAA | 14 | 4 | 0 | `null` |
| BB | 20 | 0 | 2 | 0.100 |
| BB | 21 | 1 | 1 | `null` |
| BB | 22 | 2 | 0 | `null` |

Two properties fall out of the table, and both are relied on downstream:

- `session + from_end` is constant within a symbol (`n - 1`). The two
  columns are the same position counted from opposite ends.
- **A row's label is null exactly when `from_end < horizon`.** The null
  tail is not approximately the last `h` bars, it is precisely them. This
  is what figure F2 checks: the share of non-null labels must fall to
  zero over exactly `h` positions, and a fabricated or padded tail would
  sit flat instead of stepping down.

Because `from_end` is measured per symbol, it stays correct for ETFs that
were delisted or that simply start late — each one's tail is cut relative
to its own last bar, not to the panel's last date.

**The row order is load-bearing.** `int_range` numbers *positions*, not
dates: it has no idea what is in the `date` column. It only means
"trading sessions" because section B already sorted by symbol then date.
Feed it unsorted rows and the numbering silently follows whatever order
the frame happens to be in — no error, no null, just a quietly wrong
`session`. The same positional assumption is what `shift(-horizon)`
depends on, which is why the sort happens once, up front, for both.

**The order of operations is load-bearing too.** Both columns are
computed before any row is dropped. Filter first — the null tail, the
eligibility screen, the holdout window — and the surviving rows are
renumbered from zero within whatever is left, so `from_end` would no
longer point at the real end of the series and F2's boundary profile
would measure the filter instead of the label.

Neither column reaches the parquet — section H selects three columns.
They exist to let the diagnostics in sections E and F keep counting in
trading sessions after the frame has been filtered down.

The notebook computes the arithmetic locally rather than calling
`fixed_time_horizon_labels`. That helper computes the identical quantity
but agrees only to a rounding step — enough to change the digest and
everything derived from it.

## 5. Section D — four assertions

Assertions, not printed descriptions, because all four failures are
silent and produce plausible-looking numbers.

1. **Incomplete windows are null, never valued.** Every row in the last
   `h` sessions of a symbol must be null. Catches a fabricated tail.
2. **No window spans a data gap.** Tolerance is derived, not tuned:
   `ceil(h * 7/5) + 7` calendar days — `h` sessions span roughly `7h/5`
   calendar days on a five-session week, plus a week for exchange
   holidays. This bounds the *calendar* span; it catches a hole of a
   week or more, but not one missing session, which widens the window by
   a day and stays inside tolerance. Proving exactly `h` *exchange*
   sessions needs a session calendar the notebook does not carry.
3. **No label crosses an entity boundary.** `labelled.height ==
   len(prices) - h * n_symbols`. That identity holds only if every
   symbol lost exactly `h` rows to the shift.
4. **Dtype is `Float64`.** Vacuous here by construction — the notebook
   writes continuous labels only. The defect it guards against lives in
   direction labels, where a null predicate falls through to the "down"
   class.

## 6. Figure F2 — boundary profile

Plots, for each position counted back from a symbol's last session, the
share of symbols with a non-null label. The curve must be flat at 1 and
fall to 0 over exactly the last `h` positions, with the dotted line
marking `h`.

A scalar "N valid" would show neither failure this catches: a label
masked by another label's null set, or a tail fabricated instead of
nulled (which would sit flat across the dotted line).

It reads only the null structure, never a value, so it is deliberately
**not** sealed to the development window — it describes the artifact's
shape, not its content.

## 7. Section E — the development window

```python
dev = {
    name: labels_df.with_columns(
        pl.col("timestamp")
        .shift(-horizon)
        .over("symbol")
        .alias("_label_end")
    )
    .filter(pl.col("_label_end") < HOLDOUT_START)
    .drop_nulls(name)
    for name, horizon in HORIZONS.items()
}
```

The key idea: the filter is on `_label_end`, **the date the label
resolves**, not on `timestamp`, the date it is observed. A row observed
three weeks before the holdout starts resolves *inside* the holdout, so
it is a holdout row. Filtering on the observation date looks sealed and
is not.

The seal governs what this notebook *looks at*. The parquet files
written in section H keep every row.

## 8. Figure F1 — distribution and scale

Both labels on one axis with identical bins (`np.linspace(-0.20, 0.20,
61)`), so the width comparison is visual rather than a column of
moments. The claim under test is square-root-of-horizon scaling: a
horizon-`h` label should be about `sqrt(h)` times as wide.

```python
# observed
ratio  = std[PRIMARY_LABEL] / std[VARIANT_LABEL]

# theoretical: sqrt(21/5) ≈ 2.05
theory = math.sqrt(PRIMARY_HORIZON / HORIZONS[VARIANT_LABEL])
```

Recorded result: std 0.0612 monthly vs 0.0311 weekly, a ratio of
**1.97** against a theoretical **2.05** — close, and the small shortfall
is consistent with mild mean reversion over the longer window.

## 9. Figure F4 — dispersion through time

A cross-sectional label is comparable across regimes only if the spread
the model ranks within is roughly stable; where it is not, the same IC
buys a different amount of return.

Order of operations matters and is deliberate:

1. std **across symbols on each date** (this is what a cross-sectional
   ranking model is scored on),
2. dates with fewer than `MIN_SYMBOLS_FOR_DISPERSION` symbols dropped,
3. those daily values **averaged within each year**.

Pooling every symbol-date in a year into one std would measure something
else — it would add the movement of the panel's own mean from date to
date into the spread across symbols on a date.

Recorded result: dispersion peaks at **6.7% in 2008**, against a
**3.9%** median year — about 1.7x. Far from constant.

## 10. Figure F3 — overlap and effective sample size

Daily sampling of a 21-session label means consecutive rows share 20 of
their 21 forward return intervals, so the row count wildly overstates
the information present. Two measurements from
`case_studies/utils/label_diagnostics.py`:

**`panel_autocorrelation(frame, col, max_lag=..., bar_col="session")`**
— autocorrelation at lags 1..max_lag, pooled across symbols. A pair is
kept only if both rows are the same symbol *and* their `session`
positions differ by exactly the lag, so no pair spans two entities or a
hole in the grid. The column is demeaned within symbol first: without
that, a panel whose symbols sit at different mean levels reports that
level dispersion as persistence.

**`effective_sample_size(frame, horizon=..., bar_col="session")`** —
returns `(rows, N_eff)` using Chapter 7.2's average-uniqueness
weighting, computed per symbol because concurrency is a property of one
entity's overlapping windows.

Note the convention documented in that function: a horizon-`h` label
consumes `h` return *intervals*, not `h+1` bars. Treating it as a closed
bar interval `[i, i+h]` would make consecutive labels appear to share
one interval even when they share none — and would report `N_eff = N/2`
for a one-day label that is in fact fully independent.

Both functions take `bar_col` for the same reason: `dev` holds only
non-null rows, and counting positions among survivors would make rows
either side of a missing bar look adjacent. `session` was numbered in
section C on the complete series. The distinction is vacuous for this
case study (property 3 proves the only nulls are each symbol's last `h`
sessions, and the holdout filter cuts a prefix, so `dev` holds an
unbroken run per symbol) but not for a case study whose bars can go
missing mid-series.

Recorded result: 418,362 rows carry **20,017 effective observations —
4.78%**, against the `1/21 = 4.76%` that a fully-overlapped window
implies. Autocorrelation runs from 0.942 at lag one to -0.019 at lag 21.

Both say the same thing: the sample is worth about a twentieth of its
height, and the purge gap between CV folds must be at least the horizon.

## 11. Section G — the baseline floor

One signal — raw 126-session (two-quarter) momentum, the lookback the
hypothesis names — against the primary label, with no feature
engineering. Establishing this before building features is what keeps a
later improvement honest.

Three details that decide whether the number means anything:

**Eligibility is applied.** `03_financial_features` keeps a feature row
only where `(symbol, year)` appears in `eligibility.csv`; the same
semi-join runs here. A baseline measured over a wider universe than the
features it gates is not a floor.

**IC is per-date, then averaged.** `cross_sectional_ic_series` computes
a Spearman rank correlation within each date and returns a time series;
the mean is taken over dates. Pooling all symbol-dates into one
correlation would mix a cross-sectional claim with a time-series one.
`min_obs` is set to half the median cross-section rather than a fixed
integer, so it means the same thing at a different universe size.

**The standard error is HAC-adjusted.** The IC series inherits the
label's overlap, so consecutive dates are not independent evidence.
`compute_ic_hac_stats` applies a Newey–West/Bartlett correction. The
`.sort("timestamp")` before it is required — HAC autocovariances are
meaningless over a permutation of time.

Recorded result: mean IC **0.0203**; naive t **3.77**; HAC t **1.08**, p
**0.282**. The bar a feature has to clear is the second number, and by
it, raw momentum is not distinguishable from noise.

> ## ⚠️ THIS CELL IS BROKEN AGAINST THE INSTALLED ENVIRONMENT
>
> `compute_ic_hac_stats(..., label_horizon=PRIMARY_HORIZON)`
> raises `TypeError` with `ml4t_diagnostic` 0.1.0b21 in
> `.venv`, which has no such parameter. The stored outputs came
> from a newer build. **Unresolved — see the open-issue section
> at the top of this document for the verification command and
> the two candidate fixes.** No change has been made to the
> notebook or the environment.

## 12. Section H — artifacts

```python
for label_name in LABEL_NAMES:
    record = write_artifact(
        labels_df
        .select(["timestamp", "symbol", label_name])
        .drop_nulls(),
        LABELS_DIR / f"{label_name}.parquet",
        keys=["timestamp", "symbol"],
        written_by="02_labels",
        inputs={"market_data": MARKET_DATA_DIGEST},
    )
```

Writes `labels/fwd_ret_21d.parquet` and `labels/fwd_ret_5d.parquet`,
each with a `<name>.parquet.digest.json` sidecar recording content
digest, row count, columns, keys, the writing stage, and the digest of
the price data it was built from.

Three columns only — the `from_end` and `session` bookkeeping stays in
memory. Every row is kept (the development seal governed diagnostics,
not output), minus the null tail.

Two things the notebook is explicit about:

- **The sidecars are not read anywhere yet.** Stage 02 writes them and
  the chain stops. Model runs are separately pinned to their input bytes
  by `_training_input_identity`
  (`case_studies/utils/latent_factors/case_study.py`), which hashes the
  label parquet, the feature parquets and `setup.yaml` into one
  aggregate digest.
- **No `cv_config.json` is written.** Folds are derived by
  `case_studies/utils/cv_window.py` from `setup.yaml` plus the label
  parquet's own timeline — from the artifact rather than from a second
  file describing it, which could only drift.

## 13. The audit record

Prints one block per label — anchor, horizon, resolution, overlap (`h-1`
sessions), base rate (mean and std on the development window), and which
stage consumes it. Built from the values computed above rather than
typed by hand, so it cannot describe a label different from the one
written.

---

## Downstream

`03_financial_features.py` reads whichever label is `labels.primary` in
`setup.yaml`, for its feature evaluation. The variant is written but not
consumed before modelling.

## Known limitations (as stated in the notebook)

- Close-to-close is not the backtest's next-open execution, and nothing
  here measures the gap. `16_costs` sweeps commission and half-spread,
  not the return definition.
- The universe is a fixed, backward-looking list carrying the
  survivorship bias that `01_feasibility_analysis` documents.
- The baseline is one signal at one lookback.
