# `04_model_based_features.ipynb` — cell-by-cell walkthrough

Companion notes for the ETF case study's model-based feature stage. The
notebook itself explains *why* each section exists; this document explains
*what each cell does mechanically*, which functions it calls, and where
the values come from.

This stage differs from `03` in one structural way that governs
everything else: its features come from **fitted models**, so avoiding
look-ahead is no longer a property of the arithmetic but of *where the
parameters came from*. Every model is refit per CV fold on that fold's
training window, and the output carries a `fold` column.

---

## ⚠️ OPEN ISSUES — three, none blocking

**Status: reported only. Nothing has been changed in the notebook.**

Documented here rather than fixed, because each is a judgement call about
how strict this stage should be.

### 1. The per-asset IC t-statistic ignores label overlap

Date-level features go through `robust_ic`, which uses a **stationary
bootstrap** and so accounts for temporal dependence. The per-asset
(GARCH) path does not — it computes the t-statistic by hand:

```python
"t_stat": float(np.nanmean(ics) / (np.nanstd(ics) / np.sqrt(len(ics)))),
"p_value": None,
```

That treats every daily cross-sectional IC as independent evidence. The
label is `fwd_ret_21d`, so consecutive dates share 20 of their 21 forward
sessions. **`02_labels` measured exactly this effect on this data**: the
same IC series read a naive t of 3.77 and a HAC-corrected t of 1.08, and
that walkthrough's section G exists to establish the second number as the
bar.

`garch_cond_vol` reports **t = 8.42** with a null p-value. That is the
naive statistic. Its true magnitude under an overlap-aware standard error
is materially smaller — I have not computed it here, because doing so
means refitting GARCH on 100 ETFs across 9 folds, but the direction is not
in question and the precedent is one notebook upstream.

The two IC paths in the same table therefore carry different uncertainty
semantics. The plotting cell is aware of this and says so, declining to
draw error bars for that reason. The t-column has no such guard.

### 2. There are no assertions

| notebook | `assert` statements |
|----------|--------------------|
| `01_feasibility_analysis` | 4 |
| `02_labels` | 4 |
| `03_financial_features` | 1, plus `warmup_audit` and `assert_values_agree` |
| **`04_model_based_features`** | **0** |

The "Quality Check" section prints coverage, means and standard
deviations; nothing fails. This matters more here than it would elsewhere,
because this stage's central claims are precisely the kind that fail
silently and still produce plausible columns:

- that probabilities are filtered rather than smoothed,
- that GARCH parameters came from training only,
- that no fold's features were influenced by its own validation window.

`03` has a directly applicable pattern in `assert_values_agree` — rebuild
on a truncated panel and compare. An analogue here would be: for a given
fold, extending the data past `val_end` must not change any feature value
inside the window.

### 3. `outer_coalesce` is deprecated

Used twice, for the FFD symbol joins and the date-level combine:

```python
ffd_fold = ffd_fold.join(df, on="timestamp", how="outer_coalesce")
```

Polars deprecated this in **0.20.29** in favour of
`how="full", coalesce=True`. It still works in the pinned 1.42.1, but
emits a `DeprecationWarning` that `warnings.filterwarnings("ignore")` at
the top of the notebook suppresses. It will break on a future upgrade,
silently until it does.

---

**Contents**

| # | Cell | What it does |
|---|---|---|
| 0 | Imports | `hmmlearn`, `arch`, `ffdiff`, `robust_ic` |
| 1 | Parameters | Restarts, GARCH minimum, symbol cap |
| 2 | Prices | 470,662 rows, 100 assets |
| 3 | Fold setup | 8 CV folds plus a synthetic holdout fold |
| 4 | Part 1 | Why one HMM on SPY rather than 100 |
| 5 | HMM fitting | K-means seeding, restarts, state ordering |
| 6 | Filtered probs | The forward algorithm, written out |
| 7 | Regime features | Stress probability, transition, duration |
| 8 | Illustrative fit | Full-sample HMM, for the figure only |
| 9 | Per-fold HMM | The loop that produces saved features |
| 10 | Part 2 | Fractional differencing at fixed `d` |
| 11 | Per-fold FFD | Ten reference ETFs per fold window |
| 12 | Part 3 | GARCH fit-then-filter |
| 13 | Per-fold GARCH | 100 ETFs × 9 folds |
| 14 | Part 4 | Broadcast and join into a per-fold panel |
| 15 | Quality check | Coverage and per-fold stability |
| 16 | Save | `features/model_based.parquet` |
| 17 | Evaluation | Validation periods only; two IC paths |
| 18 | Takeaways | Eight points, and what is not checked |

---

## 0. Imports and helpers

- **Aim** — pull in the three model families and the evaluation helper.
- **Shows** — that each family comes from an established library rather
  than being reimplemented: `hmmlearn` for the HMM, `arch` for GARCH,
  `ml4t.engineer` for fractional differencing.
- **Outcome** — the modelling imports, plus `generate_cv_splits` and
  `robust_ic`.

```python
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from ml4t.diagnostic.evaluation.stats import robust_ic
from ml4t.engineer.features.fdiff import ffdiff
from sklearn.cluster import KMeans
```

`warnings.filterwarnings("ignore")` is set here, which is worth noting
because it is what hides the deprecation in open issue 3 and the
convergence warnings from 900 GARCH fits.

## 1. Run parameters

- **Aim** — hold the knobs papermill overrides for CI.
- **Shows** — the cost controls: restart count and a minimum training
  length below which GARCH is not attempted.
- **Outcome** — `N_RESTARTS = 10`, `GARCH_MIN_OBS = 504`,
  `MAX_SYMBOLS = 0` meaning all.

```python
N_RESTARTS = 10
GARCH_MIN_OBS = 504   # ~2 years
MAX_SYMBOLS = 0       # 0 = all symbols
```

`GARCH_MIN_OBS = 504` is the reason some folds fit fewer than 100 ETFs —
see section 13. `MAX_SYMBOLS = 0` uses a sentinel rather than `None`,
guarded by `if MAX_SYMBOLS > 0`, which differs from `03`'s convention of
`None` for "unset".

## 2. Prices

- **Aim** — load the full price panel.
- **Shows** — that this stage reads raw prices, not `03`'s feature matrix.
- **Outcome** — 470,662 rows, 100 assets.

Note the stage reads `load_etfs()` directly and does **not** join
`financial.parquet`. The two feature sets stay separate artifacts and are
combined downstream, which is why this notebook's output has its own
`fold` column rather than extending `03`'s matrix.

## 3. Fold setup and the synthetic holdout fold

- **Aim** — obtain the walk-forward folds, then add one more covering the
  sealed period.
- **Shows** — that the holdout needs features too, and they must come from
  a model that never saw it.
- **Outcome** — 8 CV folds plus fold 8, the holdout: train
  2013-01-03..2024-01-01, validate 2024-01-01..2025-12-31.

```python
holdout_fold = {
    "fold": len(cv_splits),
    "train_start": cv_splits[0]["train_start"],
    "train_end": holdout_start,
    "val_start": holdout_start,
    "val_end": str(holdout_end),
}
all_folds = cv_splits + [holdout_fold]
```

**The folds roll rather than expand.** Fold 0 trains 2013–2022, fold 7
trains 2006–2015 — each about a ten-year window, most recent first. The
holdout fold borrows `cv_splits[0]["train_start"]`, so it inherits the
same window length rather than training on everything available.

**This fold is constructed by hand, not by the splitter.** That is a
deliberate extension — `generate_cv_splits` derives folds inside the
development window and by design will not produce one that validates on
the holdout. The seal still holds because `train_end` is exactly
`holdout_start`.

## 4. Part 1 — why one HMM on SPY

- **Aim** — justify a single market-level regime model rather than one per
  ETF.
- **Shows** — three reasons: avoiding 100 independently overfitted HMMs,
  regime being a market-level phenomenon, and cross-sectional consistency.
- **Outcome** — SPY log returns and 21-day annualized volatility, 5,010
  observations from 2006-02-02.

The consequence worth carrying forward: **every ETF inherits the same
regime state on a given date**, so these are date-level features. That
makes them useless for cross-sectional ranking — a point the evaluation in
section 17 has to work around, and the reason regime features are called
*conditioning* rather than signal.

## 5. HMM fitting — seeding, restarts and state ordering

- **Aim** — fit a 2-state Gaussian HMM reproducibly.
- **Shows** — two defences against the standard failure modes of EM: local
  optima, and label switching.
- **Outcome** — `fit_hmm_kmeans_init` and `sort_states_by_variance`.

```python
model = GaussianHMM(n_components=n_states, covariance_type="full",
                    n_iter=200, random_state=random_state,
                    init_params="st")   # only startprob and transmat
model.means_ = kmeans.cluster_centers_
model.covars_ = np.array([np.cov(X[kmeans.labels_ == k].T)
                          + np.eye(X.shape[1]) * 1e-6 for k in range(n_states)])
```

**`init_params="st"` is the load-bearing argument.** It tells `hmmlearn`
to initialise only the start probabilities and transition matrix, leaving
means and covariances for the caller — which are then set from k-means
centroids. Without it EM would overwrite the seeding.

The `+ np.eye(...) * 1e-6` ridge keeps each covariance positive-definite
when a cluster is nearly degenerate.

**Label switching is prevented by sorting on variance**, ascending, so
state 0 is always calm and state 1 always stressed. This matters because
the model is refit nine times: without a canonical ordering, EM could
return "stress" as state 0 in one fold and state 1 in the next, and
`regime_prob_stress` would flip sign across the fold boundary while
looking entirely well-formed.

## 6. Filtered probabilities — the forward algorithm, written out

- **Aim** — obtain `P(z_t | x_1:t)` rather than `P(z_t | x_1:T)`.
- **Shows** — why `hmmlearn`'s own `predict_proba` cannot be used.
- **Outcome** — `compute_filtered_probs`, a log-domain forward pass.

This is the single most important cell in the notebook. `predict_proba`
returns **smoothed** probabilities, conditioned on the *entire* sequence
including future observations. Using them as features would leak the
future into every row — not through the parameters, but through the
inference.

```python
fwdlattice[0] = log_startprob + framelogprob[0]
for t in range(1, n_samples):
    for j in range(n_components):
        fwdlattice[t, j] = framelogprob[t, j] + np.logaddexp.reduce(
            fwdlattice[t - 1] + log_transmat[:, j])
```

The forward recursion runs entirely in the log domain and normalises at
the end, which is what keeps it stable over 3,000-step sequences where the
raw likelihood underflows. `np.log(... + 1e-300)` guards a zero transition
probability.

## 7. Deriving the three regime features

- **Aim** — turn a filtered posterior into features a model can use.
- **Shows** — a subtle causality trap in the duration feature, and how it
  is avoided.
- **Outcome** — `regime_prob_stress`, `regime_transition`,
  `regime_log_duration`.

```python
regime_prob_stress = filtered_sorted[:, 1]
states_sorted = (regime_prob_stress >= 0.5).astype(int)   # NOT model.predict
```

**The state used for duration is the argmax of the filtered posterior, not
the Viterbi path.** The docstring is explicit about why, and it is the
sharpest observation in the notebook: `model.predict` returns the *global
MAP sequence* over the whole window, so a validation-period state would
depend on observations after it. `regime_log_duration` would then encode
the future in a column that looks like a simple run length.

Note this makes the notebook internally inconsistent by design: the
illustrative figure in section 8 *does* use `model.predict`, because there
the point is the retrospective picture rather than a feature.

`regime_transition` is the absolute one-day change in stress probability,
with `prepend` so the first element is zero rather than dropped.
`regime_log_duration` is `log1p` of the run length, compressing a
heavy-tailed count.

## 8. The illustrative full-sample fit

- **Aim** — draw one regime overlay so a reader can see whether the model
  finds known episodes.
- **Shows** — the 2-state split: calm 69.0% of the time at 10.9%
  volatility, stressed 31.0% at 27.3%.
- **Outcome** — a figure only. **These features are not saved.**

```text
Illustrative full-sample HMM: best log-likelihood = -22419.8
State 0 (Low-Vol): 69.0% of time, mean ret=0.072%, vol=10.9%
State 1 (High-Vol): 31.0% of time, mean ret=-0.030%, vol=27.3%
```

**This model is fit on the entire sample, holdout included**, and uses
Viterbi states for the shading. The notebook disclaims it in bold and the
disclaimer is accurate — nothing from this fit reaches `hmm_features`. It
is worth being aware of anyway: the figure a reader forms their intuition
from is the one object here that saw the sealed period.

The state separation is the useful readout: a mean return of +0.072% in
calm against −0.030% in stress, at two and a half times the volatility, is
what makes "stress" a meaningful label rather than an arbitrary partition.

## 9. The per-fold HMM loop

- **Aim** — produce the regime features that are actually saved.
- **Shows** — fit on training, filter over the whole fold window.
- **Outcome** — 25,382 rows across 9 folds.

```python
spy_train = spy_full.filter((timestamp >= train_start) & (timestamp < train_end))
# ... N_RESTARTS fits, keep best log-likelihood ...
spy_fold  = spy_full.filter((timestamp >= train_start) & (timestamp <= val_end))
filtered  = compute_filtered_probs(best_fold_model, X_fold)
```

**Note the asymmetric bounds**: training is `< train_end` while the fold
window is `<= val_end`. Parameters come from strictly-before the boundary;
inference runs through it.

Per-fold mean stress ranges from 0.233 (fold 6) to 0.391 (fold 3), which
is the expected variation from ten-year windows covering different
regimes — fold 3 trains 2009–2019, fold 6 trains 2006–2016.

The loop `continue`s on insufficient data or a failed fit, so a fold could
silently be absent from the output. The printed line is the only signal,
and nothing downstream asserts that all nine folds are present.

## 10. Part 2 — fractional differencing at fixed `d`

- **Aim** — obtain stationary series that retain long-range memory.
- **Shows** — why `d` is fixed by asset class rather than estimated.
- **Outcome** — ten reference ETFs, `d = 0.4` for equities and gold,
  `0.5` for fixed income.

**Fixing `d` is a look-ahead argument, not a convenience.** Estimating `d`
— by minimising an ADF statistic, say — would be a data-dependent
optimisation, and doing it on the full sample would leak. Pre-specifying
by asset class makes the transform purely mechanical, so the only reason
to compute it per fold is warmup, not fitting.

The ten ETFs span US large/tech/small cap, developed and emerging
international, long treasuries, gold, real estate, high yield and
investment grade — a deliberate cross-asset spread rather than the whole
universe.

## 11. The per-fold FFD loop

- **Aim** — apply the transform inside each fold window.
- **Shows** — that warmup is handled per fold so no fold inherits another's
  leading values.
- **Outcome** — 10 series, 25,403 rows across folds.

```python
log_close = etf["close"].log()
ffd_series = ffdiff(log_close, d=d)
ffd_df = pl.DataFrame({...}).drop_nulls()
```

The FFD filter is a fixed-width weighted sum of past values, so it needs a
warmup stretch and nothing else. Restarting it at each `train_start` costs
a little data and guarantees the series inside a fold depends only on that
fold's window.

`ffd_hyg` is the one column with materially fewer values — 2,397,251
against 2,423,866 — because HYG lists later than the earliest fold's
`train_start`.

This is where `outer_coalesce` appears (open issue 3): the ten per-symbol
frames are joined on timestamp, so a symbol with a longer warmup
contributes nulls rather than truncating the others.

## 12. Part 3 — GARCH fit-then-filter

- **Aim** — get a conditional volatility path per ETF without refitting
  inside the validation window.
- **Shows** — the `fix()` idiom: estimate on training, then run the
  variance recursion forward with frozen parameters.
- **Outcome** — `fit_garch_fold`.

```python
train_result = train_model.fit(disp="off", show_warning=False)
full_model = arch_model(full_returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="Normal")
filtered = full_model.fix(train_result.params)
cond_vol_ann = filtered.conditional_volatility * np.sqrt(252) / 100
```

**`fix()` is what makes this causal.** It applies given parameters to a
series without estimating anything, so `σ_t` depends only on returns up to
`t` given those parameters. Refitting on the full window would put
validation-period returns into the parameters; not extending the
recursion at all would leave the validation window without features.

Returns are scaled by 100 before fitting — `arch` is numerically happier
on percent returns — and the resulting volatility is divided back out and
annualised by `sqrt(252)`.

## 13. The per-fold GARCH loop

- **Aim** — run that fit across every ETF and fold.
- **Shows** — where the minimum-observation floor bites.
- **Outcome** — 2,420,952 rows, 100 assets, 9 folds; conditional
  volatility mean 0.177, std 0.123.

Fitted counts per fold: 100, 100, 100, 99, 99, 99, 98, 98, 100. **The
misses are all in the older folds**, where an ETF had not yet accumulated
`GARCH_MIN_OBS = 504` training sessions. A symbol that fails is simply
absent, so its `garch_cond_vol` is null after the join in section 14 —
skipped, never imputed, which is the right choice but is not asserted
anywhere.

This is the expensive cell: 900 model fits.

## 14. Part 4 — combine and broadcast

- **Aim** — assemble one `(fold, timestamp, symbol)` panel.
- **Shows** — date-level features broadcast to every symbol; per-asset
  features joined on both keys.
- **Outcome** — 14 columns, 2,423,866 rows, 100 assets, 9 folds.

```python
panel = fold_skeleton.join(date_level, on="timestamp", how="left")
panel = panel.join(fold_garch, on=["timestamp", "symbol"], how="left")
```

**The row count deserves attention.** The underlying price panel has
470,662 rows; this output has 2,423,866 — roughly five times as many.
Nothing is duplicated in error: each `(timestamp, symbol)` pair appears
**once per fold whose window covers it**, carrying that fold's model
output, and the fold windows overlap heavily by construction.

The practical consequence: **any downstream consumer must filter on
`fold`**. Joining this to labels without doing so would multiply every row
several times over and silently inflate every count and every regression.
The notebook's own evaluation in section 17 does filter correctly.

Both joins are `how="left"` from the skeleton, so a missing model output
becomes null rather than dropping the row.

## 15. Quality check and per-fold stability

- **Aim** — confirm coverage and that features do not drift wildly across
  folds.
- **Shows** — valid counts, means and standard deviations per column, then
  means per fold.
- **Outcome** — all 14 columns above 98.9% coverage; fold means varying
  smoothly.

```text
regime_prob_stress : 2,422,627/2,423,866 valid, mean=0.3194, std=0.4522
ffd_hyg            : 2,397,251/2,423,866 valid, mean=0.0958, std=0.0947
garch_cond_vol     : 2,420,952/2,423,866 valid, mean=0.1772, std=0.1234
```

`regime_prob_stress` has a standard deviation (0.452) larger than its mean
(0.319), which for a quantity bounded in [0,1] means it is strongly
bimodal — the filtered posterior spends most of its time near 0 or near 1
rather than hedging. That is characteristic of a well-separated two-state
fit and consistent with the volatility split in section 8.

**This section prints; it does not assert** — see open issue 2. A column
that came back entirely null, or a fold that failed to fit, would show up
here as a number a reader has to notice.

## 16. Save artifacts

- **Aim** — write the per-fold panel.
- **Shows** — a plain parquet write.
- **Outcome** — `features/model_based.parquet`, 2,423,866 rows, 14
  features plus the fold column.

```python
temporal.write_parquet(FEATURES_DIR / "model_based.parquet")
```

**No digest sidecar.** `02_labels` and `03_financial_features` both write
through `write_artifact`, which records a content digest and the digests
of the inputs. This stage calls `write_parquet` directly, so its output
carries no provenance and two vintages of it are indistinguishable
downstream. Given that this artifact depends on nine fitted models with
stochastic restarts, it is arguably the one that most needs a digest.

## 17. Incremental evaluation

- **Aim** — read whether any of these features carry signal, without
  touching the seal.
- **Shows** — the holdout fold excluded, and two different IC methods for
  two different feature shapes.
- **Outcome** — an evaluation panel of 200,977 rows; a 14-row IC table.

**The scoping is careful and correct.** Only the validation periods of the
eight CV folds are used, and the holdout fold is deliberately excluded —
its features are saved for final evaluation, but letting it inform
feature-quality expectations here would break the seal.

**Two IC paths, because the features have two shapes:**

- **Per-asset (`garch_cond_vol`)** — cross-sectional Spearman IC within
  each date, averaged over dates.
- **Date-level (HMM, FFD)** — these are identical across symbols on a
  date, so their cross-sectional IC is zero by construction. They are
  instead correlated against the *cross-sectional average* forward return,
  through `robust_ic`.

Selected results:

| feature | IC | t | p |
|---------|-----|---|---|
| `regime_prob_stress` | 0.200 | 2.40 | 0.011 |
| `ffd_tlt` | 0.185 | 2.45 | 0.001 |
| `garch_cond_vol` | 0.067 | **8.42** | *null* |
| `ffd_iwm` | −0.187 | −2.67 | 0.004 |
| `ffd_spy` | −0.201 | −3.06 | — |

Two things to read carefully here.

**The equity FFD features are all negative and the bond ones positive**,
at similar magnitudes. They are not seven independent findings — the
transform is applied to the log price level, so these largely re-express
one market-direction variable, split by asset class. `03`'s redundancy
clustering would be the natural check and is not run on this set.

**`garch_cond_vol`'s t of 8.42 is not comparable to the others** — see
open issue 1. It is the naive statistic on an overlapping label; every
other row in the table has a bootstrap standard error behind it. The
plotting cell explicitly refuses to draw error bars because the two
uncertainty scales differ, which is the right instinct, but the t-column
in the printed table carries no equivalent warning.

## 18. Key takeaways, and what is not checked

- **Aim** — state the eight points the notebook draws out.
- **Shows** — each as a look-ahead avoided rather than a technique
  applied.
- **Outcome** — the handoff to the Chapter 11 models.

The eight are: per-fold fitting removes parameter look-ahead; one HMM on
the aggregate market; filtered rather than smoothed probabilities; state
sorting to prevent label switching; GARCH fit-then-filter via
`model.fix()`; fixed `d` for fractional differencing; per-ETF GARCH giving
cross-sectional differentiation; and a dedicated holdout fold.

Reading them against the rest of the pipeline, three things this stage
does *not* do stand out — none fatal, all noted above:

- **It asserts nothing.** Its causal claims are exactly the ones that fail
  quietly, and `03` already has the pattern that would test them.
- **It writes no digest.** The one artifact here built from stochastic
  fits is also the one with no provenance record.
- **It runs no redundancy check**, though ten of its fourteen columns are
  transforms of price levels that plainly move together.

**Next**: the Chapter 11 models join `financial.parquet` with
`model_based.parquet` and use the `fold` column so each CV fold sees only
features fitted on its own training window.
