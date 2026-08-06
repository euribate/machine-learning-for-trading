# `02_labels.ipynb` — cell-by-cell walkthrough

Companion notes for the ETF case study's label-engineering stage. The
notebook itself explains *why* each section exists; this document
explains *what each cell does mechanically*, which functions it calls,
and where the values come from.

Section 0 is documented in depth — it binds the notebook to
`config/setup.yaml`, and the resolution chain behind it is not obvious
from reading the line. Everything else is covered at normal depth.

---

## ✅ RESOLVED — section G's `TypeError` on re-run

**Status: fixed in the notebook. The environment was not changed.**

Section G called `compute_ic_hac_stats(ic, ic_col="ic",
label_horizon=PRIMARY_HORIZON)`. The `ml4t_diagnostic` pinned in `.venv`
(**0.1.0b21**) has no such parameter and takes no `**kwargs`, so the call
raised `TypeError: unexpected keyword argument 'label_horizon'`. The
stored outputs had come from a newer build.

**The fix.** One line, no environment change:

```python
stats = compute_ic_hac_stats(ic, ic_col="ic", maxlags=PRIMARY_HORIZON - 1)
```

**Why `- 1`, and why that is not a guess.** Reading `0.1.0b25` — the
build that has the parameter — its lag helper is:

```python
def _newey_west_lag(n, horizon=None):
    nw_auto = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    base = max(int(horizon) - 1, nw_auto) if horizon is not None else nw_auto
    return max(1, min(base, max(1, n // 2)))
```

So `label_horizon=h` resolves to `max(h - 1, Newey–West auto)`, capped at
`T // 2`. With `h = 21` and `T = 4,257` IC dates the auto rule gives 9, so
the effective lag is **20** — not 21. An earlier draft of this document
proposed `maxlags=PRIMARY_HORIZON`; that was off by one and is why the
`- 1` is written explicitly here.

The rationale is a real statistical one rather than a magic number:
overlapping `h`-session labels induce MA(`h-1`) dependence in the daily IC
series, so the bandwidth has to reach `h-1` lags. The sample-size rule
alone does not.

**Verified against the recorded outputs**, on the actual panel:

| call | lags | mean IC | HAC t | p |
|------|------|---------|-------|---|
| `maxlags=PRIMARY_HORIZON - 1` | 20 | 0.0203 | **1.08** | **0.282** |
| `maxlags=PRIMARY_HORIZON` | 21 | 0.0203 | 1.07 | 0.287 |
| no lag argument (auto) | 9 | 0.0203 | 1.36 | 0.175 |

The first row reproduces the notebook's stored numbers exactly. The third
shows what was at stake: left to the sample-size rule, the baseline's
t-statistic reads 1.36 at p 0.175 — still short of significance, but a
visibly different floor from the one the commentary describes.

Cross-checked across both builds on a synthetic MA(20) series:
`label_horizon=21` on `0.1.0b25` and `maxlags=20` on `0.1.0b21` agree to
fifteen significant figures, which also establishes that nothing else in
the HAC computation changed between the two builds.

**Upgrading instead** — `uv pip install ml4t-diagnostic==0.1.0b25` — is
equally valid and lets the original call stand. It was not chosen here
because it bumps a package every other chapter imports, to buy a
statistic this one line already produces. Note that `0.1.0b25` warns when
called *without* `label_horizon`, which is worth knowing if the pin ever
moves.

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
| 11 | Section G | Baseline IC floor — HAC bandwidth set from the horizon |
| 12 | Section H | Write parquet + digest sidecars |
| 13 | Audit record | Printed label definition table |

---

## 0. The configuration cell (in depth)

- **Aim** — bind the notebook to `config/setup.yaml` so the label horizon
  has exactly one source of truth across labelling, cross-validation and
  the backtest.
- **Shows** — where the number `21` actually comes from: a three-source
  resolution chain that falls through to the CV *buffer* field, which is
  not the same concept as a horizon.
- **Outcome** — two label names and two horizons. Every loop in the rest
  of the notebook iterates that dict.

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

The notebook needs the integer `21` for `pl.col("close").shift(-horizon)`.
The only real question is where `21` comes from — typed into the notebook,
or resolved from config.

Typing it creates a **second copy of the number**. The rest of the
pipeline — cross-validation windows (`case_studies/utils/cv_window.py`),
the backtest, stage 03's feature evaluation — reads the horizon from
`setup.yaml`. Change the strategy to a two-month hold, edit `setup.yaml`,
and the notebook would keep labelling at 21 while the CV purge gap moved
to 42. No exception, no warning, just a leak. Resolving keeps **one
source of truth**.

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

They are usually equal and here they are — both 21, which is why reading
one from the other works. They need not be: you might use a 5-day horizon
but purge 21 days because a *feature* has a 21-day lookback that would
leak. Different jobs, separate fields — as the notebook's markdown says.

**The risk this flags:** with source 2 absent, the horizon is read out of
the buffer field. Set `buffer: 30D` later as a safety margin while still
intending a 21-day label, and this cell silently builds **30-day
labels**. An explicit `horizons:` block closes that hole.

### `.rstrip("Dd")` and `int(...)`

`setup.yaml` stores durations with a unit suffix (`21D`, `5D`), the
format the backtest and CV code also consume, so
`"21D" → rstrip("Dd") → "21" → int() → 21`.

`rstrip` takes a **set of characters**, not a suffix, stripping any of
them repeatedly from the right — so `"5d"` and `"21DD"` work too. It also
strips blindly: `21W` (weeks) or `3M` (months) survive intact and then
explode on `int()`. It handles the day convention only.

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

- **Aim** — resolve the case-study root from the repo layout rather than
  from the notebook's own location.
- **Shows** — which four families of dependency the stage draws on.
- **Outcome** — `CASE_DIR` and `LABELS_DIR`, identical whether the
  notebook runs under Jupyter or papermill.

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

- **Aim** — expose the knobs papermill overrides when the notebook runs
  in CI rather than in production.
- **Shows** — why shortening a CI run is done by date and never by
  thinning the universe.
- **Outcome** — three constants; production leaves the first two `None`.

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

- **Aim** — get the price panel into the one row order every later shift
  depends on, and fingerprint the exact bytes it was built from.
- **Shows** — why no eligibility filter is applied at this point, even
  though one exists and is applied later.
- **Outcome** — `prices`, sorted by symbol then timestamp, plus
  `MARKET_DATA_DIGEST`, which section H records as every label's input.

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

- **Aim** — turn the price panel into the forward-return labels the rest
  of the pipeline trains on, and record each row's position in its
  symbol's history before any later step destroys it.
- **Shows** — that the arithmetic stays inside one symbol, that an
  incomplete forward window is `null` rather than a number, and that the
  resulting null tail is exactly the last `h` bars of each symbol.
- **Outcome** — `labels_df`: the price panel plus `fwd_ret_21d`,
  `fwd_ret_5d`, and the `from_end` / `session` bookkeeping columns. This
  is the frame every diagnostic below reads and section H writes from.

### The label itself

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
convention. Two decisions in that expression carry the section:

- **`.over("symbol")`** confines the shift inside each ETF, so no window
  reaches across an entity boundary into an unrelated price. Section D's
  third assertion is what proves it actually held.
- **The shift yields `null`** where a full forward window does not exist.
  That is the honest encoding of "unknown". Writing `0` instead would
  claim a flat return the data never observed, and would survive every
  downstream mean, correlation and IC as if it were evidence.

### The two bookkeeping columns

```python
labels_df = prices.with_columns(
    (pl.len().over("symbol") - 1 - pl.int_range(pl.len()).over("symbol")).alias("from_end"),
    pl.int_range(pl.len()).over("symbol").alias("session"),
)
```

**Why they exist.** Both record where a row sat in its symbol's own
history, while that is still knowable. Everything downstream drops rows —
the null tail, the eligibility screen, the holdout cut — and survivors
renumber from zero. So position is stamped on now, on the complete
series. The two columns are the same position anchored to opposite ends,
because two diagnostics need different ends.

**How they are built.** `.over("symbol")` is SQL's
`OVER (PARTITION BY symbol)`: evaluate once per symbol, scatter results
back to their own rows, shape and order preserved. Inside it `pl.len()`
is *that symbol's* row count, so `pl.int_range(pl.len())` yields
`0 … n-1` down the group — that is `session`. A bare
`pl.len().over("symbol")` instead broadcasts `n` onto every row, and
`n - 1 - session` flips the count around, so `from_end` is `0` on the
symbol's last bar.

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

Two properties fall out, both relied on downstream: `session + from_end`
is constant within a symbol (`n - 1`), and **a label is null exactly when
`from_end < horizon`** — precisely the last `h` bars, not approximately.

**`from_end` is anchored to the last bar**, because the null tail is
defined by the end. Section D's first assertion filters `from_end <
horizon` to select the rows that must be null, and figure F2 groups on it
to check the tail steps down over exactly `h` positions rather than
sitting flat. Measured per symbol, so ETFs that were delisted or started
late are cut relative to their own last bar, not the panel's last date.

**`session` is anchored to the first bar**, because section F needs a
time axis and passes it as `bar_col="session"`. Lags there must be in
trading sessions — the horizon is — and calendar days are not uniform
across weekends and holidays. It also cannot be row position within
`dev`, which holds only surviving rows: counting among survivors would
make rows either side of a missing bar look adjacent.

**Row order is load-bearing.** `int_range` numbers *positions*, not
dates. It means "trading sessions" only because section B already sorted
by symbol then date — the same assumption `shift(-horizon)` rests on.
Unsorted input yields a silently wrong `session`: no error, no null.

Neither column reaches the parquet; section H selects three.

The notebook computes the arithmetic locally rather than calling
`fixed_time_horizon_labels`. That helper computes the identical quantity
but agrees only to a rounding step — enough to change the digest and
everything derived from it.

## 5. Section D — four assertions

- **Aim** — make the label's correctness a condition of the notebook
  running, rather than something a reader is asked to take on trust.
- **Shows** — four ways a forward-return label fails *silently*: it
  produces a full column of plausible numbers, and every downstream mean,
  IC and t-statistic keeps working on top of the damage.
- **Outcome** — the run halts on any breach. On success it prints, per
  label, the labelled row count, the observed calendar span range, and
  the tolerance it was checked against.

Assertions rather than printed descriptions, because a printed number
only helps a reader who already knows what it should be. Each assertion
below names the defect it exists to catch.

```python
tol = math.ceil(horizon * 7 / 5) + 7
tail = spanned.filter(pl.col("from_end") < horizon)
labelled = spanned.drop_nulls(label_name)
```

**1. An incomplete window is null, never valued.** `from_end < horizon`
selects exactly the rows whose forward window runs off the end of their
symbol (section C), and every one must be null. Catches a fabricated or
padded tail — the failure where the last `h` bars carry invented returns
that look like ordinary data.

**2. No window spans a data gap.** The tolerance is derived, not tuned:
`ceil(h * 7/5) + 7` calendar days, since `h` sessions span roughly `7h/5`
calendar days on a five-session week, plus a week for exchange holidays.
**Its reach is limited and the notebook says so.** It bounds the
*calendar* span, so it catches a hole of a week or more but not a single
missing session, which widens the window by one day and stays inside
tolerance. Proving exactly `h` *exchange* sessions would need a session
calendar this notebook does not carry.

**3. No label crosses an entity boundary.** Checked by identity rather
than by inspection:

```python
labelled.height == len(prices) - horizon * prices["symbol"].n_unique()
```

Every symbol must lose exactly `horizon` rows to the shift and no others.
If one window had closed over the boundary into the next symbol, that
symbol would keep a row it should have lost and the count would not
balance. This is the assertion that makes `.over("symbol")` in section C
a verified claim instead of an intention.

**4. The dtype is `Float64`.** **Vacuous here by construction** — this
notebook writes continuous labels only, so the check cannot fail. It is
carried because the defect it guards against is real in the direction
labels of later stages, where a null predicate falls through to the
"down" class and turns missing data into a confident bearish call.

Two of the four are therefore weaker than they look: assertion 2 is
partial, assertion 4 currently proves nothing. Assertions 1 and 3 are the
ones doing load-bearing work here.

## 6. Figure F2 — boundary profile

- **Aim** — inspect the *shape* of the null tail, which no single count
  can describe.
- **Shows** — the share of symbols carrying a non-null label at each
  position counted back from their last session.
- **Outcome** — a curve flat at 1 that falls to 0 over exactly the last
  `h` positions. Anything else is a defect.

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

- **Aim** — seal this notebook's own diagnostics so nothing it reports is
  informed by the holdout period.
- **Shows** — that a label must be sealed on the date it *resolves*, not
  the date it is observed; the two differ by the horizon.
- **Outcome** — a `dev` frame per label, used by every figure below.
  The parquet files in section H are unaffected and keep every row.

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

- **Aim** — check the two labels are the same quantity measured over
  different horizons, not two unrelated constructions.
- **Shows** — both distributions on one axis with identical bins, tested
  against square-root-of-horizon scaling.
- **Outcome** — a ratio of 1.97 against a theoretical 2.05.

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

- **Aim** — establish whether the label's cross-sectional spread is
  stable enough that one IC means the same thing in every regime.
- **Shows** — dispersion across symbols per date, averaged by year, with
  the order of operations chosen so it measures spread and not drift.
- **Outcome** — a 6.7% peak in 2008 against a 3.9% median year. Not
  constant, and worth remembering when reading a single IC.

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

- **Aim** — find out what the row count is actually worth once daily
  sampling of a 21-session label is priced in.
- **Shows** — how fast the overlap decays, and the same panel re-counted
  under average-uniqueness weighting.
- **Outcome** — 418,362 rows carry 20,017 effective observations. Sets
  the floor on the CV purge gap.

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

Both take `bar_col` for the reason given in section C: `dev` holds only
non-null rows, so counting positions among survivors would make rows
either side of a missing bar look adjacent. The distinction is vacuous
here — property 3 proves the only nulls are each symbol's last `h`
sessions, and the holdout cuts a prefix, so `dev` holds an unbroken run
per symbol — but not for a case study whose bars go missing mid-series.

Recorded result: 418,362 rows carry **20,017 effective observations —
4.78%**, against the `1/21 = 4.76%` that a fully-overlapped window
implies. Autocorrelation runs from 0.942 at lag one to -0.019 at lag 21.

Both say the same thing: the sample is worth about a twentieth of its
height, and the purge gap between CV folds must be at least the horizon.

## 11. Section G — the baseline floor

- **Aim** — record what a trivial signal already achieves, before any
  feature engineering, so a later improvement can be judged against it.
- **Shows** — the three choices that decide whether the number means
  anything: eligibility, per-date IC, and an overlap-aware standard error.
- **Outcome** — mean IC 0.0203 with a HAC t of 1.08. The floor is
  indistinguishable from noise.

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

**The standard error is HAC-adjusted, at a bandwidth set by the
horizon.** The IC series inherits the label's overlap, so consecutive
dates are not independent evidence:

```python
ic = cross_sectional_ic_series(...).sort("timestamp")
stats = compute_ic_hac_stats(ic, ic_col="ic", maxlags=PRIMARY_HORIZON - 1)
```

`compute_ic_hac_stats` applies a Newey–West/Bartlett correction, and two
things about that line are load-bearing:

- **`.sort("timestamp")` is required.** HAC autocovariances are
  meaningless over a permutation of time, and nothing would raise if the
  frame arrived unsorted.
- **The lag is `h - 1` = 20, not the sample-size default.** A 21-session
  label overlapping daily induces MA(20) dependence in the IC series, so
  the bandwidth has to reach 20 lags to price it in. Left to the
  Newey–West rule `floor(4 * (T/100)^(2/9))`, the bandwidth here would be
  **9**, and the same data would report a t of 1.36 at p 0.175 instead of
  1.08 at p 0.282 — a materially more flattering floor, from a standard
  error that has not fully accounted for the overlap this notebook spent
  section F measuring.

Recorded result: mean IC **0.0203**; naive t **3.77**; HAC t **1.08**, p
**0.282**. The bar a feature has to clear is the second number, and by
it, raw momentum is not distinguishable from noise.

The three standard errors are worth seeing together, since they are the
same data under three assumptions about independence:

| standard error | lags | t | p |
|----------------|------|---|---|
| naive (dates independent) | — | 3.77 | — |
| HAC, sample-size bandwidth | 9 | 1.36 | 0.175 |
| HAC, horizon bandwidth | 20 | **1.08** | **0.282** |

The gap between the first and last row is the entire practical
consequence of overlap: a signal that looks decisively significant
becomes indistinguishable from noise once the evidence is counted
correctly.

## 12. Section H — artifacts

- **Aim** — write the labels to disk together with enough provenance to
  tell one run from another.
- **Shows** — what is deliberately excluded: bookkeeping columns, and any
  second file describing the folds.
- **Outcome** — two parquet files and their digest sidecars.

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

- **Aim** — leave a written definition of each label that cannot drift
  from the label actually written.
- **Shows** — anchor, horizon, resolution, overlap, base rate and
  consumer, all derived from the computed values rather than typed.
- **Outcome** — one printed block per label.

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
