# `01_feasibility_analysis.ipynb` — cell-by-cell walkthrough

Companion notes for the ETF case study's feasibility stage. The notebook
itself explains *why* each section exists; this document explains *what
each cell does mechanically*, which functions it calls, and where the
values come from.

This stage fits nothing. It asks whether `config/setup.yaml` describes a
strategy the data can actually support, and writes one artifact —
`eligibility.csv` — that every later stage joins against.

---

**Contents**

| # | Cell | What it does |
|---|---|---|
| 0 | Imports | Modules and the three feasibility helpers |
| 1 | Parameters | Date range and the dollar-volume floor |
| 2 | Configuration | Reads every knob out of `setup.yaml` |
| 3 | Section A | States the three questions being asked |
| 4 | Section B.1 | Load prices, assert two properties |
| 5 | Section B.2 | Point-in-time eligibility, breadth at each decision |
| 6 | Section B.3 | Round-trip cost per fund, exceedance curve |
| 7 | Section B.4 | Within-fund autocorrelation of the carrier |
| 8 | Section B.5 | Move scale against cost, as numbers |
| 9 | Section C | Design decisions and kill conditions |
| 10 | Section D.1 | Decision count vs row count |
| 11 | Section D.2 | Fold generation and the purge gap |
| 12 | Section E | Writes `eligibility.csv` |
| 13 | Section F | Findings tabulated against `setup.yaml` |
| 14 | Takeaways | Three rules, three limitations |

---

## 0. Imports and helpers

- **Aim** — pull in the three panel diagnostics whose correct version is
  longer than a notebook cell should be.
- **Shows** — which work is delegated: the statistics live in
  `case_studies/utils/feasibility.py`, the loading and interpretation stay
  in the notebook.
- **Outcome** — `exceedance_curve`, `fold_timeline` and `panel_acf`, plus
  the loader, the splitter and the shared plot style.

```python
from case_studies.utils.feasibility import exceedance_curve, fold_timeline, panel_acf
from data import load_etfs
from utils.cv_splits import generate_cv_splits
from utils.paths import get_case_study_dir
from utils.style import COLORS, FIGSIZE, add_message_title
```

Each of the three helpers exists because the naive version of its
statistic is wrong on a panel — the module docstring says so directly.
Sections 5, 7 and 11 give the specific failure each one avoids.

## 1. Run parameters

- **Aim** — hold the two numbers that are not in `setup.yaml`.
- **Shows** — that the eligibility *rule* is declared in config while its
  *threshold* is not, which is a gap in the config rather than a choice.
- **Outcome** — the date range and `ADV_THRESHOLD = 10e6`.

```python
CASE_STUDY_ID = "etfs"
START_DATE = "2006-01-01"
END_DATE = "2025-12-31"
ADV_THRESHOLD = 10e6
```

The notebook's own configuration markdown flags the reason:
`universe.eligibility_rule` "declares the rule without its number, so the
dollar-volume floor is declared in the parameters cell." Everything else
is resolved from config in section 2. This one constant is the exception,
and it decides who is in the universe.

## 2. Configuration — reading `setup.yaml`

- **Aim** — bind every knob the notebook tests to the file that declares
  it, so the analysis cannot drift from the strategy it is validating.
- **Shows** — that the numbers under test are read, never retyped; the
  notebook is checking a specification it does not get to restate.
- **Outcome** — the development/holdout boundary, 100 declared assets, a
  breadth floor of 20, cost parameters, and horizons `[5, 21]`.

```python
BREADTH_FLOOR = max(SETUP["backtest"]["sweep"]["top_k_grid"][PRIMARY_LABEL])
HORIZONS = sorted(int(re.search(r"(\d+)d$", n).group(1)) for n in LABELS)
```

Two derivations worth reading closely:

**`BREADTH_FLOOR` is the largest position count the sweep asks for**, not
a round number. The backtest sweeps a grid of `top_k` values; the
universe must be able to fill the widest book in that grid on every
decision date, so the floor is `max(top_k_grid)` — here **20**. Widen the
grid in `setup.yaml` and the bar this notebook holds breadth to rises
with it, automatically.

**Horizons are parsed out of the label names** with `(\d+)d$`, rather
than read from a horizon field. That works because the names encode the
number (`fwd_ret_21d` → 21), and it is a different resolution path from
the one `02_labels` uses, which goes through `resolve_label_horizon` and
falls back to the CV buffer fields. Two stages, two ways of learning the
same number. A label named without a trailing `<digits>d` would raise
`AttributeError` on the `re.search(...).group(1)` here.

Printed output:

```text
Development 2006-01-01 to 2024-01-01 | sealed holdout to 2025-12-31
100 declared, floor 20 | horizons [5, 21] sessions
```

## 3. Section A — orientation

- **Aim** — state what a cross-sectional ETF rotation needs from its data,
  before any of it is measured.
- **Shows** — that the strategy trades *differences between funds' recent
  paths*, so it needs many funds quoting at once more than a few quoting
  well. That distinction sets what the rest of section B measures.
- **Outcome** — three questions, which map onto B.2, B.3/B.5 and D.

Markdown only. The three questions are: does the universe exist on every
decision date, is a typical move large next to what it costs to capture,
and are there enough decision dates for a clean walk-forward. Sections 5
through 11 answer them in that order.

## 4. Section B.1 — load and verify the universe

- **Aim** — load the price panel and establish two properties before
  anything is computed from it.
- **Shows** — that the loaded symbols are a subset of `universe.assets`
  and that no close is at or below zero.
- **Outcome** — `prices` (full range) and `research` (development window
  only): 100 funds, 420,462 rows, last session 2023-12-29.

```python
research = prices.filter(pl.col("timestamp") < pl.lit(HOLDOUT_START).str.to_date())

undeclared = sorted(set(research["symbol"].unique().to_list()) - DECLARED_ASSETS)
assert not undeclared, f"loaded but absent from setup.yaml::universe.assets: {undeclared}"
assert research["close"].min() > 0, "a non-positive close is not a denominator"
```

**Two frames, and the difference matters throughout.** `prices` spans the
full range including the holdout; `research` is cut at `HOLDOUT_START`.
Every figure and statistic below reads `research`. The one deliberate
exception is the eligibility table in section 5 — see the note there.

The `close > 0` assertion is not defensive boilerplate: `close` becomes a
denominator twice below, in the cost-in-bps conversion and in
`pct_change`. The assertion message says exactly that.

The set difference is one-directional. It catches a symbol that was
loaded but never declared; it does *not* catch a declared symbol that
failed to load, which would silently shrink the universe. The count in
the printed line (`100 funds`) is what a reader has to check that
against.

## 5. Section B.2 — breadth at every decision date

- **Aim** — decide universe membership on information the strategy would
  actually have had, then count it on the dates the strategy acts.
- **Shows** — that a whole-sample liquidity filter would keep precisely
  the funds that *stayed* liquid, and that one universe count hides
  whether enough funds are eligible on any particular decision date.
- **Outcome** — the `eligibility` table, and a breadth series that clears
  the floor of 20 from the second year onward.

```python
eligibility = (
    prices.with_columns(
        dollar_volume=pl.col("close") * pl.col("volume"), year=pl.col("timestamp").dt.year()
    )
    .group_by(["symbol", "year"])
    .agg(pl.col("dollar_volume").mean().alias("adv"), pl.len().alias("n_days"))
    .filter((pl.col("n_days") >= 200) & (pl.col("adv") >= ADV_THRESHOLD))
    .select("symbol", (pl.col("year") + 1).alias("eligible_year"))
    .unique()
    .sort(["symbol", "eligible_year"])
)
```

**`year + 1` is the whole point-in-time mechanism.** Volume is aggregated
for calendar year `Y`, and the row that survives grants eligibility for
year `Y + 1`. A fund is therefore admitted on what was already known when
the year opened. Drop that `+ 1` and the rule would admit funds on the
strength of volume they had not yet traded.

**`n_days >= 200`** excludes a fund that quoted for only part of a year,
which has no annual average to be admitted on. A fund listing in October
would otherwise be judged on a two-month mean.

**This cell reads `prices`, not `research`** — the only computation in
section B that does. That is deliberate and stated in section 12: the
artifact must cover the sealed years too, because downstream stages join
against it there. It is not leakage, because eligibility for year `Y`
reads only year `Y - 1` volume, so the sealed years' rows are derived
from sealed years' own history. It is worth noticing anyway, since the
configuration markdown's blanket claim that "Section B computes on the
development window alone" is true of every figure but not of this table.

Breadth is then counted at month-ends:

```python
month_end = research.filter(
    pl.col("timestamp") == pl.col("timestamp").max().over(pl.col("timestamp").dt.truncate("1mo"))
)
breadth = (
    month_end.with_columns(year=pl.col("timestamp").dt.year())
    .join(eligible, ["symbol", "year"], how="left")
    .group_by("timestamp")
    .agg(pl.col("eligible").fill_null(False).sum().alias("n_eligible"))
    .sort("timestamp")
)
```

**The left join is load-bearing**, and the notebook comments it inline: a
left join keeps the dates on which *nothing* is eligible, which an inner
join would drop from the frame entirely. Those dates are exactly the
failure the chart exists to show — an inner join would produce a picture
with the bad dates silently missing and a floor that appears always
cleared.

`month_end` picks the last session of each calendar month by comparing
each timestamp to the max within its own truncated month, so it selects
real sessions rather than calendar month-ends that may not have traded.

## 6. Section B.3 — the round trip, and what a move is worth

- **Aim** — price the round trip per fund, then read the size of a
  typical move against the cost of capturing it.
- **Shows** — that a per-share cost is a *different* cost on every fund,
  so one cost line drawn on raw returns answers the question for no fund
  in particular.
- **Outcome** — `COST_BPS` (universe median 9.45 bps) and an exceedance
  curve on which break-even sits at exactly 1.

```python
half_spread = pl.col("symbol").replace_strict(
    HALF_SPREADS, default=DEFAULT_HALF_SPREAD, return_dtype=pl.Float64
)
cost = (
    research.group_by("symbol")
    .agg(pl.col("close").median().alias("price"))
    .with_columns((2 * (half_spread + PER_SHARE) / pl.col("price") * 1e4).alias("cost_bps"))
    .sort("cost_bps")
)
```

The `2 *` is the round trip — two half-spreads and two commissions, in
and out. Dividing by price converts dollars per share into a fraction of
the position, and `* 1e4` puts it in basis points. Because both inputs are
dollars per share, **the same cost is a smaller fraction on a
higher-priced fund**, which is why the resulting bar chart spans an order
of magnitude across one universe.

`replace_strict` with an explicit `default` assigns the tier spread where
`setup.yaml` names one and the default elsewhere. The notebook is candid
that this is an assumption, not a measurement: daily bars carry no bid and
ask, so the spread is asserted, and `18_cost_sensitivity` stresses it.

The exceedance curve then scales every move by its own fund's cost:

```python
returns = research.with_columns(
    cost_bps=2 * (half_spread + PER_SHARE) / pl.col("close") * 1e4
).with_columns(
    (pl.col("close").pct_change(h).abs() * 1e4 / pl.col("cost_bps").shift(h))
    .over("symbol")
    .alias(f"h{h}")
    for h in HORIZONS
)
```

Three details decide whether that ratio means anything:

- **`cost_bps` is recomputed per row here**, on each session's close,
  rather than reusing the per-fund median from the bar chart. The chart
  summarises; this measures each move against the cost prevailing at the
  time.
- **`.shift(h)`** takes the cost at the price the position *opened* at,
  `h` sessions before the move being measured. Using the closing price's
  cost would price the trade at an outcome that had not happened yet.
- **`.abs()`** makes this a magnitude, not a return. The curve says
  nothing about direction and therefore nothing about profit — a point
  section 8 repeats deliberately.

Dividing by cost puts break-even at **1** on a scale every fund shares,
which is what makes a single vertical line at `x = 1` legible for a
hundred funds at once. `exceedance_curve` returns the survival function
of the magnitudes, thinned to 400 quantiles so a multi-million-row panel
draws as a curve rather than a raster.

The curve runs over every session in the development window, not only the
decision dates — the notebook is explicit that this is the scale of a
move against cost, not an opportunity set.

## 7. Section B.4 — serial correlation of the carrier

- **Aim** — measure how much of one month's return the next month
  repeats, in the series the ranking is actually built from.
- **Shows** — the difference between a within-entity autocorrelation and
  one computed over a stacked panel, which would measure the joins
  between funds rather than persistence within them.
- **Outcome** — a mean within-fund ACF over 12 lags with a
  10th–90th-percentile band: a single month's return says almost nothing
  about the next month's.

```python
monthly = month_end.with_columns(monthly_return=pl.col("close").pct_change().over("symbol"))
acf = panel_acf(monthly, entity_col="symbol", value_col="monthly_return", max_lags=12).filter(
    pl.col("lag") > 0
)
```

**The carrier is the month-to-month return**, computed on `month_end` —
the same dates the strategy acts on — not on daily closes. Persistence at
the daily frequency would not be the property a monthly rotation depends
on.

`panel_acf` computes an ACF per fund and returns the cross-entity mean,
the 10th and 90th percentiles of the per-fund curves, and a white-noise
band of `1.96 / sqrt(mean series length)`. Funds with fewer than
`max(min_obs, max_lags + 1)` observations are skipped, so short-lived
funds do not contribute a noisy curve.

`lag > 0` drops lag zero, which is a series against itself and equals 1 by
construction; the notebook's inline comment notes its bar would flatten
every other one.

This is a property of each fund's own series, not of the cross-sectional
ranking — the notebook points at `05_evaluation` for the latter. A weak
result here does not sink the strategy: a rotation ranks funds against
each other, and that is a different question from whether any one fund
repeats itself.

## 8. Section B.5 — move scale against cost

- **Aim** — reduce the exceedance curve to two numbers that can be
  checked against a threshold.
- **Shows** — the median move as a multiple of the median round trip, and
  the share of moves clearing their own fund's cost.
- **Outcome** — cost 0.90–38.91 bps (median 9.45); median 21-session move
  287.2 bps, a ratio of **30x**; **0.967** of moves clear.

```python
moves = returns.select(
    move_bps=1e4 * pl.col("close").pct_change(PRIMARY_HORIZON).abs().over("symbol"),
    clears=pl.col(f"h{PRIMARY_HORIZON}") > 1,
)
```

The two numbers are computed differently on purpose. The **ratio** uses
two medians — the median move over the median cost, a universe-level
summary. The **clearance share** compares each move to *its own fund's*
cost, so a fund whose cost is ten times the median is judged against its
own bar rather than the universe's.

**Both are unsigned magnitudes.** The notebook says so twice, and the
point survives repeating: a ratio of 30x does not mean a strategy earns
30x its costs. It means cost is not the thing that would stop one. Whether
a signal can predict the *direction* of those moves is Chapter 8's
question, and section 9's first kill condition.

## 9. Section C — design decisions

- **Aim** — record the three choices `setup.yaml` makes and tie each to
  the evidence that motivates it.
- **Shows** — that each kill condition is tested where its evidence lives,
  not asserted here.
- **Outcome** — cadence, kill conditions and mapping class, each with a
  named chapter that tests it.

Markdown only. The three:

- **Cadence** — monthly, ranking at the month-end close and executing at
  the next open. B.3 supports rebalancing at least that often, so cost is
  not what sets the cadence. Monthly also buys a purge gap the width of
  the primary label; the weekly horizon stays in `labels.variants` so the
  shorter holding period is measured rather than assumed away.
- **Kill conditions** — an IC indistinguishable from zero at every
  lookback (Chapter 8); a move-to-cost ratio under one once realistic
  costs are charged, or an equal-weight book beating the strategy on
  Sharpe at a smaller drawdown (both Chapter 16).
- **Mapping class** — long only, because many of these funds are
  expensive or impossible to borrow, so a short leg would price that
  constraint rather than the signal. Equal weight, because an optimised
  weighting folds in a covariance estimate and leaves the ranking's own
  contribution unidentifiable. Chapter 17 sweeps the alternatives.

Note that the kill conditions are written to be falsifiable *before* the
evidence exists — which is what makes them kill conditions rather than
post-hoc commentary.

## 10. Section D.1 — effective sample size

- **Aim** — establish what a cross-sectional evaluation actually has to
  spend.
- **Shows** — that the row count and the decision count differ by more
  than an order of magnitude, because folds are cut on the session
  timeline while the strategy acts only at month-ends.
- **Outcome** — 4,529 sessions, **216 decision dates**, 77 eligible funds
  per decision.

```text
Sessions 4,529 | decision dates 216 | eligible funds per decision 77
```

216 is the number that matters: eighteen years of month-ends. Folds are
cut on the session timeline — the same one `04_model_based_features`
hands the splitter — but a monthly strategy only gets 216 chances to act,
and a walk-forward has to divide *those* among its folds.

## 11. Section D.2 — fold demonstration

- **Aim** — generate the declared folds and confirm they fit the
  development window.
- **Shows** — that a purge gap the width of the label horizon sits between
  each training and validation block, and that no fold reaches into the
  holdout.
- **Outcome** — 8 folds, last validation ending 2023-11-29, both asserted
  rather than eyeballed.

```python
splits = generate_cv_splits(
    research.select("timestamp"),
    case_study_id=CASE_STUDY_ID,
    label_buffer=LABEL_BUFFER,
    date_col="timestamp",
)
last_val = max(s["val_end"] for s in splits)
assert len(splits) == SETUP["evaluation"]["n_splits"], "fold count differs from setup.yaml"
assert last_val < np.datetime64(HOLDOUT_START), "a fold reaches into the holdout"
```

The two assertions check different failures: the first that the splitter
produced the fold count `setup.yaml` declares, the second that the seal
actually held. Neither is a print statement, so neither can be skimmed
past.

**The figure draws the splitter's own output.** `fold_timeline` takes the
boundary dictionaries `generate_cv_splits` returned and draws those,
rather than re-deriving folds from a timeline. Its docstring records why
this matters concretely: `plot_cv_folds` re-splits whatever timeline it is
handed, and on a timeline truncated at the last validation date it
disagreed with the reported folds **by eleven days**. Drawing the
boundaries themselves means the figure and the folds cannot come apart.

The purge gap is drawn as the span between `train_end` and `val_start`,
so it is visible as a real interval rather than implied.

## 12. Section E — the eligibility artifact

- **Aim** — write the one thing this notebook hands downstream.
- **Shows** — that the artifact is a fund-year membership table, not a
  filtered price panel.
- **Outcome** — `eligibility.csv`, **1,670 fund-year pairs**.

```python
eligibility.write_csv(CASE_DIR / "eligibility.csv")
```

`02_labels` and `03_financial_features` semi-join on this table, so a
fund contributes rows only in years it cleared the floor. It covers the
sealed years too — as section 5 explains, prior-year volume is all the
rule ever reads, so extending it across the holdout introduces no
information from inside the holdout.

Two columns only: `symbol` and `eligible_year`. The table is a claim about
membership, and it deliberately carries no prices, so a downstream stage
cannot accidentally read market data through it.

## 13. Section F — findings vs `setup.yaml`

- **Aim** — put each knob next to the evidence that supports it and the
  condition that would revise it.
- **Shows** — the failure that would force each parameter to change,
  written down before it happens.
- **Outcome** — a four-row table, plus a printed reconciliation.

```text
universe.n_assets 100, eligible per decision date 0 to 96, under the floor on 12 of 216 dates
decision.cadence monthly_month_end | labels.primary fwd_ret_21d
evaluation.n_splits 8, generated 8, last validation ends 2023-11-29, holdout untouched
```

**The `0` is not a defect.** Breadth starts at zero in the first year,
because no fund yet has a prior year to be admitted on — the `year + 1`
rule from section 5 showing its edge. All 12 of the 216 dates below the
floor of 20 fall in that first year.

`n_splits 8, generated 8` is the reconciliation that matters: the number
declared and the number produced, printed side by side rather than
assumed equal.

## 14. Key takeaways and limitations

- **Aim** — state the three transferable rules, and the three places this
  analysis is weaker than it looks.
- **Shows** — each rule as a failure avoided, not a technique applied.
- **Outcome** — the handoff to `02_labels`.

The three rules: decide membership on prior-year information and count it
on the decision date; convert a per-share cost into bps before comparing
it to a return, and scale each move by its own fund's round trip; compute
a panel autocorrelation inside each entity, never across the stacked
panel.

The limitations are stated by the notebook and worth carrying forward:

- **The universe list itself is survivorship-biased.** The funds in
  `universe.assets` were chosen knowing which of them still trade, so the
  point-in-time filter removes a bias *inside* the list and not the bias
  *in* the list. This is the sharpest of the three — the machinery in
  section 5 is careful, and it cannot fix the selection that preceded it.
- **Adjusted closes distort both inputs.** `close` is adjusted for splits
  and distributions, so early prices sit below what a fund traded at:
  dollar volume is understated there, and the round trip — dollars per
  share over a price — is overstated. The half-spread is by tier, and the
  floor is not inflation-adjusted.
- **Eligibility is annual while decisions are monthly.** A fund turning
  illiquid in March keeps its place until January.

**Next**: `02_labels` builds labels at the declared horizons on this
development window, and semi-joins the eligibility table written here.
