# Macro Regimes Notebook - Operative Guide

**Notebook**: `01_process_is_edge/macro_regimes.ipynb`
**Book reference**: Chapter 1, Section 1.4 "Market Regimes: Change Is the Constant"
**Companion**: Follows `factor_regimes.ipynb` (style-factor view); this is the
macro-indicator view.
**Last verified**: 2026-07-02 (FRED data through Dec 2025, S&P 500 through Dec 2025)

---

## Table of Contents

1. [Purpose and Context](#1-purpose-and-context)
2. [Models and Techniques Used](#2-models-and-techniques-used)
3. [Step-by-Step Code Walkthrough](#3-step-by-step-code-walkthrough)
4. [Comment Alignment Audit](#4-comment-alignment-audit)
5. [How to Interpret the Results](#5-how-to-interpret-the-results)
6. [Key Caveats](#6-key-caveats)

---

## 1. Purpose and Context

This notebook detects market regimes using **macroeconomic indicators** from FRED,
then validates the detected regimes against realized S&P 500 volatility and
drawdowns.

### How it relates to factor_regimes

| | factor_regimes | macro_regimes |
|--|---------------|---------------|
| **Input** | 9 factor return series (AQR) | 4 macro indicators (FRED) |
| **History** | 99 years (1927-2026) | 24 years (2002-2026) |
| **Regime count** | 2 (Risk-On / Risk-Off) | 4 (Expansion / Recovery / Crisis / Tightening) |
| **Validation** | Factor returns by regime | S&P 500 volatility and drawdown |
| **Key insight** | Factor dynamics shift across regimes | Macro regimes map to volatility environments |

### Why macro indicators?

Factor returns (used in `factor_regimes`) are available only at the same
frequency as trading returns. Macro indicators — unemployment, interest rates,
inflation, yield curve — move at a slower pace and are published with known
schedules. They reflect the *economic environment* rather than asset-level
dynamics, making them complementary inputs for regime detection.

### Key insight

The notebook's central finding: macro regimes line up with distinct **volatility**
environments more cleanly than they line up with average returns. This makes
macro indicators useful for **risk management** (anticipating volatility shifts)
rather than return prediction.

### Scope: descriptive, NOT predictive

Like `factor_regimes`, the GMM and scaler are fitted on the full sample.
Labels are assigned ex-post. Using these labels as trading features would
constitute look-ahead bias.

---

## 2. Models and Techniques Used

### 2.1 Gaussian Mixture Model (GMM) — 4 components

Same model as in `factor_regimes` (see that guide for full GMM background).
Key difference: here K=4 is chosen *a priori* to match the Two Sigma approach
and to capture four recognized economic phases (Expansion, Recovery, Crisis,
Tightening). No BIC/AIC grid search is performed — the goal is
interpretability against known economic narratives, not statistical optimality.

Parameters: `covariance_type="full"`, `n_init=10`, `reg_covar=1e-6`,
`random_state=42`.

### 2.2 K-Means (comparison)

Used on the extended dataset as a comparison. K-Means produces hard
assignments (each month belongs to exactly one cluster), while GMM provides
soft probability assignments that quantify regime uncertainty.

### 2.3 Hierarchical (Agglomerative) Clustering

Ward linkage is used to cluster:
- **Features** (correlation clustermap): reveals which indicators group
  together (e.g., labour/yield-curve block vs. short-rate block).
- **Observations** (dendrogram): shows how months cluster, with cophenetic
  correlation measuring how faithfully the tree represents pairwise distances.

**Cophenetic correlation**: Measures the correlation between the original
pairwise distances and the distances implied by the dendrogram. Values above
0.7 indicate a reliable tree structure.

### 2.4 PCA (Principal Component Analysis)

Applied to the 25-indicator extended dataset to reduce dimensionality before
clustering. PCA finds orthogonal axes of maximum variance. The first 4
components capture 93.6% of variance, suggesting that the 25 indicators
are largely driven by ~4 underlying economic factors.

### 2.5 StandardScaler and `scale()`

- **Core analysis**: StandardScaler (zero mean, unit variance) on 4 indicators.
- **Extended analysis**: `sklearn.preprocessing.scale()` applied column-wise
  (same effect as StandardScaler but applied via `DataFrame.apply`).

### 2.6 CPI Level → YoY Change

The raw CPIAUCSL series is a non-stationary price level that trends upward
over time. Clustering on it would produce clusters that separate "early years
vs. recent years" rather than "inflationary vs. non-inflationary." The notebook
converts it to year-over-year percentage change (`pct_change(12) * 100`),
which is stationary and captures inflationary regimes.

---

## 3. Step-by-Step Code Walkthrough

The notebook follows a two-part structure. **Part 1** builds a simple regime
model from 4 hand-picked macro indicators, validates it against the S&P 500,
and then examines the correlation structure of those indicators. **Part 2**
asks: *can we do better with more data?* — expanding to 25 indicators,
comparing multiple clustering methods, and testing dimensionality reduction.

### Part 1: Core Analysis (4 Indicators)

> **Goal**: Build a minimal, interpretable regime model and verify that the
> clusters correspond to distinct market environments.

#### Steps 1-4: Data Preparation

**Step 1 — Load FRED data**:

```python
macro_raw = load_macro()
```

The `load_macro()` function reads the FRED parquet file downloaded via
`data/macro/download.py` (requires `FRED_API_KEY` in `.env`).
Output: 9,497 daily rows, 25 series, 2000-01-01 to 2025-12-31.

**Step 2 — Select core indicators**: Four indicators are chosen to cover the
main dimensions of the macro environment:

| Indicator | What it measures | Why included |
|-----------|-----------------|-------------|
| UNRATE | Unemployment rate (%) | Labor market health — rises in recessions |
| DFF | Federal Funds effective rate (%) | Monetary policy stance — the Fed's primary tool |
| T10Y2Y | 10Y-2Y Treasury spread | Yield curve slope — classic recession predictor |
| CPIAUCSL | Consumer Price Index (level) | Inflation — converted to YoY % change |

These four cover labour market, monetary policy, term structure, and prices.
They are available monthly with minimal lags and have long, reliable histories.

**Step 3 — Resample to monthly**: Daily FRED data is resampled by taking the
last observation in each month. Forward-fill handles release lags (e.g., CPI
publishes mid-month for the prior month). A backward-fill catches leading
nulls — this introduces a small look-ahead at the panel boundary, acceptable
for a descriptive demo. The data is filtered to 2002+ where all 4 series have
reliable coverage.

Result: 289 raw monthly rows → 277 months after CPI YoY transformation
(loses 12 months for the lagged percentage change). Date range: Jan 2003 to
Jan 2026.

**Step 4 — Transform CPI and standardize**:

```python
macro_df["cpi_yoy"] = macro_df["cpiaucsl"].pct_change(12) * 100
macro_scaled = StandardScaler().fit_transform(macro_df)
```

The 4 working features after this step: `unrate`, `dff`, `t10y2y`, `cpi_yoy`.

Actual descriptive statistics (verified):

| Stat | unrate | dff | t10y2y | cpi_yoy |
|------|--------|-----|--------|---------|
| Mean | 5.75 | 1.76 | 1.07 | 2.58 |
| Std | 2.02 | 1.91 | 0.97 | 1.81 |
| Min | 3.40 | 0.04 | -1.06 | -1.96 |
| Max | 14.80 | 5.41 | 2.84 | 8.98 |

At this point the data is clean, stationary, and on a common scale. The next
step fits the clustering model.

#### Steps 5-6: Fit GMM and Label Regimes

**Step 5 — Fit GMM with K=4**:

```python
gmm_macro = GaussianMixture(n_components=4, ...)
```

K=4 is fixed *a priori* to match the Two Sigma approach and to capture four
recognized economic phases (Expansion, Recovery, Crisis, Tightening). No
BIC/AIC grid search is performed — the goal is interpretability against known
economic narratives, not statistical optimality.

Silhouette score: **0.252** (reasonable structure; above 0.25 threshold).

**Step 6 — Regime labeling**: The GMM assigns arbitrary numeric labels (0-3).
The code examines the mean values of each indicator per cluster, then applies a
priority cascade of rules to assign interpretive names:

Actual regime means (verified):

| Cluster | unrate | dff | t10y2y | cpi_yoy | Label |
|---------|--------|-----|--------|---------|-------|
| 0 | 4.70 | 1.36 | 1.12 | 2.15 | Expansion |
| 1 | 12.30 | 0.07 | 0.47 | 0.56 | Crisis |
| 2 | 7.62 | 0.11 | 1.88 | 1.58 | Recovery |
| 3 | 4.43 | 3.79 | 0.24 | 4.01 | Tightening |

Labeling rules (priority order):
1. unrate > 10 → **Crisis** (cluster 1: COVID-era unemployment spike)
2. unrate > 6 and dff < 0.5 → **Recovery** (cluster 2: high unemployment
   but Fed at zero = post-crisis healing)
3. dff > 3 and t10y2y < 0.5 → **Tightening** (cluster 3: Fed hiking,
   curve flattening)
4. cpi_yoy > 4 → Inflation (not triggered here)
5. unrate < 5 and dff < 2 → **Expansion** (cluster 0: low unemployment,
   moderate rates)
6. Fallback → Transition

The model has produced four clusters with economically coherent profiles. But
do these clusters actually correspond to different market conditions? The next
steps test that.

#### Steps 7-9: Validating the Regimes Against the Market

The core question: **do the macro clusters correspond to meaningfully different
market environments?** If the regimes are real, they should map to distinct
volatility and drawdown profiles in the S&P 500.

**Steps 7-8 — S&P 500 validation**: The S&P 500 daily index is loaded,
resampled to monthly, aligned to the macro regime dates, and split by regime.
For each regime, the code computes annualized volatility and maximum drawdown.

Actual regime statistics (verified):

| Regime | Months | Ann. Vol (%) | Max DD (%) |
|--------|--------|-------------|-----------|
| Expansion | 55 | 12.3 | 13.8 |
| Recovery | 98 | 15.2 | 50.8 |
| Crisis | 4 | 16.0 | 9.2 |
| Tightening | 82 | 16.0 | 40.5 |

**Verdict**: Volatility rises monotonically from Expansion (12.3%) through
Recovery (15.2%) to Crisis/Tightening (16.0%). The regimes capture genuinely
different risk environments, confirming the model is picking up real economic
structure, not noise.

Note: the Crisis regime has only 4 months (COVID peak) — too few for reliable
statistics. Its 9.2% max drawdown is misleadingly low because the drawdown
calculation resets within each regime's non-contiguous months. The true COVID
drawdown (~34%) spans months that the GMM splits across Crisis and Recovery.

**Step 9 — Regime timeline visualization (Figure 1.6)**: A multi-panel figure
summarizes the results:
- **Left**: Swim lanes per regime (filled when active), with event markers
  at 2008 (GFC), 2020 (COVID), 2022 (Inflation).
- **Right columns**: Horizontal bar charts showing annualized volatility
  and max drawdown per regime.

Regimes are sorted from lowest to highest volatility (Expansion at top,
Crisis/Tightening at bottom). Artifacts are persisted to
`output/macro_regimes/figure_1_6/inputs.npz` for the book's publication-quality
figure generator.

At this point the core model is built and validated. The remaining step in
Part 1 examines *why* joint clustering works and *what the model might be
missing*.

#### Step 10: Understanding the Indicator Relationships

Before expanding to more indicators, the notebook pauses to examine how the
four core indicators relate to each other. This serves two purposes:

1. **Justifies the GMM approach**: If the indicators were uncorrelated,
   K-Means would suffice. The correlation heatmap shows they are not.
2. **Exposes the model's blind spots**: If some economic dimensions are
   missing or redundant in the 4-indicator set, that motivates expanding to
   a broader panel in Part 2.

Actual correlations (verified):

| | dff | t10y2y | unrate | cpi_yoy |
|--|-----|--------|--------|---------|
| dff | 1.00 | -0.73 | -0.59 | 0.33 |
| t10y2y | -0.73 | 1.00 | 0.70 | -0.40 |
| unrate | -0.59 | 0.70 | 1.00 | -0.43 |
| cpi_yoy | 0.33 | -0.40 | -0.43 | 1.00 |

**Key relationships**:
- **UNRATE ↔ T10Y2Y (+0.70)**: When unemployment rises, the yield curve
  steepens as the Fed cuts the front end.
- **DFF ↔ T10Y2Y (-0.73)**: Hiking cycles flatten or invert the curve.
- **CPI YoY** is largely orthogonal to the other three — inflation regimes
  can coexist with both recession and expansion.

**Implication**: Three of the four indicators (UNRATE, DFF, T10Y2Y) are
strongly correlated — they partially capture the same underlying dynamic
(the Fed's response to the business cycle). CPI brings an independent
dimension, but it is the *only* additional axis. This suggests the
4-indicator model may be information-starved: its modest silhouette of 0.252
could improve if more independent economic dimensions were included.

This leads directly to Part 2.

### Part 2: Extended Analysis (25 Indicators)

> **Goal**: Test whether a richer set of indicators produces better-separated
> regimes and whether alternative clustering methods confirm the core model's
> findings.

Part 1 showed that the 4-indicator model works (the regimes map to distinct
volatility environments) but is limited: three of the four indicators are
highly correlated, and the silhouette of 0.252 is modest. Part 2 investigates
three questions:

1. **More data**: Do 25 indicators produce better clusters than 4?
2. **Alternative methods**: Do K-Means and hierarchical clustering agree
   with GMM?
3. **Dimensionality**: Can PCA compress 25 indicators without losing cluster
   quality?

#### Step 11-12: Exploring the Extended Dataset

**Step 11 — Prepare full dataset**: All 25 FRED series with <50% missing data
are selected. They span interest rates (DFF, DGS1-DGS30), yield curve spreads,
VIX, initial claims (ICSA), Fed balance sheet (WALCL), CPI, core CPI, PCE,
unemployment, payrolls, labor participation, industrial production, M2, GDP,
and derived yield curve slopes.

Result: 277 months, 25 series (standardized with `scale()`).

**Step 12 — Visualize all series**: Before clustering, the notebook plots all
25 standardized time series on a grid. This is an exploratory step — it lets
you visually identify which indicators carry regime information and which are
noise. VIX, ICSA, and DFF show the most visible regime structure (sharp spikes
and level shifts), while slow-moving series (M2, housing) carry less regime
information.

With the data explored, the next step examines the internal correlation
structure to understand how these 25 indicators relate to each other.

#### Step 13: Hierarchical Clustering of Features

A `seaborn.clustermap` groups the 25 indicators by correlation similarity
(Ward linkage). This answers: **which indicators carry redundant information,
and which bring independent dimensions?**

Four blocks emerge:
- **Labour / yield-curve block** (CIVPART, UNRATE, T10Y2Y) — the business
  cycle, confirming the high correlations seen in Step 10.
- **Stress block** (VIX, ICSA) — market fear and layoffs, absent from the
  core model.
- **Growth / price-level block** (INDPRO, CPI, M2) — real economy and
  inflation.
- **Short-rate block** (DFF, DGS1-DGS3) — monetary policy at different
  maturities.

This confirms that the 25 indicators are *not* 25 independent signals — they
cluster into ~4 thematic groups. It also reveals what the core model was
missing: the stress dimension (VIX, initial claims) and the growth dimension
(industrial production, M2) were not represented in the 4-indicator set.

#### Step 14: GMM vs K-Means on Extended Data

Now the central test: do the richer data and alternative methods confirm
the core model's regime structure?

Both GMM and K-Means are fitted with K=4 on the 25-indicator dataset.

Actual silhouette scores (verified):

| Model | Silhouette |
|-------|-----------|
| GMM | 0.417 |
| K-Means | 0.448 |

Both are substantially higher than the core 4-indicator GMM (0.252),
confirming that the extended panel captures more cluster structure. K-Means
edges out GMM slightly on silhouette, but silhouette is a distance-based
metric that naturally favors K-Means's spherical clusters. GMM provides
probability assignments that K-Means cannot — and these matter.

The GMM probability heatmap shows soft transitions between regimes (gradual
color changes), while the K-Means heatmap shows hard switches (binary on/off).
The soft transitions are more realistic — economic conditions don't snap
between states overnight.

#### Step 15: Hierarchical Clustering of Observations

Step 13 clustered the *features* (columns) to see which indicators group
together. This step clusters the *observations* (rows — the 277 months) using
Ward linkage to see whether a completely different algorithm (agglomerative,
bottom-up) agrees with GMM's top-down partitioning.

The dendrogram visualizes how months merge into clusters at increasing
distance thresholds.

**Cophenetic correlation**: 0.710 — above the 0.7 quality threshold,
confirming that the hierarchical tree faithfully represents the pairwise
distance structure. The fact that a third, structurally different clustering
method also finds coherent groupings reinforces confidence in the regime
structure.

#### Step 16: Can PCA Reduce Dimensionality Without Losing Cluster Quality?

With 25 indicators, the GMM must estimate a large number of covariance
parameters. PCA can compress the data into fewer orthogonal dimensions,
potentially reducing noise and speeding up fitting — but at the risk of
discarding cluster-relevant signal.

PCA on the 25-indicator dataset:

| Component | Cumulative Variance |
|-----------|-------------------|
| PC1 | 45.6% |
| PC2 | 80.4% |
| PC3 | 87.9% |
| PC4 | 93.6% |
| PC5 | 96.0% |

Just 4 components capture 93.6% of variance — consistent with the 4 thematic
blocks found by hierarchical clustering in Step 13. The 25 indicators are
largely driven by ~4 underlying economic factors.

GMM fitted on the 10-component PCA reduction achieves silhouette **0.396** —
lower than raw (0.417), suggesting that some of the discarded variance
(~4%) contains cluster-relevant signal. PCA helps with interpretability
(fewer dimensions to reason about) but does not improve separation.

#### Step 17: Final Comparison — Core vs Extended

The notebook concludes by comparing all three approaches side by side:

Actual silhouette comparison (verified):

| Model | Silhouette |
|-------|-----------|
| Core (4 indicators) | 0.252 |
| Extended (25 indicators) | 0.417 |
| Extended + PCA (10 components) | 0.396 |

The extended panel provides substantially better separation (+65% over core),
but the core model is more interpretable. The trade-off depends on the use
case:
- **Narrative / communication**: Use the 4-indicator model — easy to explain
  ("unemployment is high, rates are low → Recovery").
- **Quantitative risk model**: Use the 25-indicator model — better cluster
  quality justifies the complexity.
- **Middle ground**: PCA on 25 indicators gives nearly as good separation
  (0.396) with fewer dimensions, but the principal components lack intuitive
  economic meaning.

---

## 4. Comment Alignment Audit

The notebook's inline comments were verified against actual outputs from the
2026-07-02 run. Most numbers match the cell outputs. Issues found:

### Key Takeaways Cell

| Comment claims | Actual value | Status |
|---------------|--------------|--------|
| "12.0% (Expansion, 77 months)" | 12.3%, 55 months | **Misaligned** |
| "16.0% (Crisis, 4 months)" | 16.0%, 4 months | OK |
| "Tightening at 15.0%" | 16.0% | **Misaligned** |
| "Recovery at 15.2%" | 15.2% | OK |
| "2002-2024 panel" | 2003-2026 panel | **Misaligned** |
| "silhouette 0.42 vs 0.45" | 0.417 vs 0.448 | OK (rounded) |
| "silhouette 0.25" (core) | 0.252 | OK (rounded) |

### Correlation Heatmap Interpretation Cell

| Comment claims | Actual value | Status |
|---------------|--------------|--------|
| "correlation ~0.70" (UNRATE ↔ T10Y2Y) | 0.70 | OK |
| "correlation ~−0.73" (DFF ↔ T10Y2Y) | -0.73 | OK |

### Qualitative Assessment

All qualitative conclusions remain correct:
- Macro regimes do map to distinct volatility environments
- Extended indicators do outperform core on silhouette
- Cophenetic correlation is above 0.7
- CPI YoY is largely orthogonal to other indicators

The numeric misalignments in the Key Takeaways are moderate: Expansion shows
55 months (not 77) and 12.3% vol (not 12.0%), and Tightening shows 16.0% vol
(not 15.0%). These likely reflect a dataset update (FRED data now extends to
Dec 2025, adding ~12 months since the comments were written).

---

## 5. How to Interpret the Results

### 5.1 The Four Regimes

| Regime | Economic story | Typical indicators |
|--------|---------------|-------------------|
| **Expansion** | Low unemployment, moderate rates, healthy growth | UNRATE < 5%, DFF ~1.4%, positive yield curve |
| **Recovery** | High unemployment, zero rates, steep curve | UNRATE > 6%, DFF near 0%, T10Y2Y > 1.5% |
| **Crisis** | Very high unemployment, emergency rates | UNRATE > 10%, DFF near 0%, flat curve |
| **Tightening** | Low unemployment, high rates, flat/inverted curve | UNRATE < 5%, DFF > 3%, T10Y2Y < 0.5% |

### 5.2 Volatility, Not Returns

The central insight is that macro regimes predict **volatility environments**,
not returns. Expansion has 12.3% annualized vol; the other three regimes
cluster around 15-16%. This means:

- A portfolio manager can use macro regime signals to **adjust risk budgets**
  (tighter stops and smaller positions during Recovery/Tightening).
- But macro regimes do **not** tell you whether equities will go up or down
  — Recovery has 50.8% max drawdown (the 2008-2009 GFC) but also includes
  the subsequent bull run.

### 5.3 Core vs Extended

The 4-indicator model (silhouette 0.252) produces interpretable regimes
directly mappable to economic narratives. The 25-indicator model (silhouette
0.417) produces better-separated clusters but the regime labels become harder
to explain. The PCA reduction (silhouette 0.396) is a middle ground: fewer
dimensions, nearly as good separation, but the principal components lack
intuitive economic meaning.

### 5.4 How This Differs from factor_regimes

| | factor_regimes | macro_regimes |
|--|---------------|---------------|
| **Signal source** | Asset returns | Economic indicators |
| **Regime duration** | Short (avg 4.3 months) | Long (avg 55-98 months) |
| **Transition frequency** | Noisy (275 transitions in 99 years) | Stable (a few per decade) |
| **Use case** | Short-term factor allocation | Medium-term risk budgeting |

Macro regimes are more stable because economic conditions evolve slowly.
This makes them more actionable than the factor-based regimes, which switch
every few months.

---

## 6. Key Caveats

1. **Short history**: 277 months (2003-2026) covers only ~2 full business
   cycles. The 4-regime model has just 4 months in Crisis — too few for
   reliable regime-level statistics.

2. **Look-ahead bias**: Full-sample fitting. The backward-fill on leading
   nulls introduces a small additional look-ahead at the panel boundary.

3. **Regime labeling is heuristic**: The `create_regime_labels()` function
   uses hardcoded thresholds (e.g., unrate > 10 → Crisis) tuned for the
   2002-2025 US economy. Different thresholds or a different country would
   produce different labels.

4. **Publication lag**: FRED indicators are published with delays (CPI:
   ~2 weeks after month-end, UNRATE: ~1 week, GDP: ~1 month). A real-time
   regime model must account for these lags; this notebook does not.

5. **S&P 500 coverage**: The S&P 500 CSV starts in 2006, not 2002. The
   regime validation statistics (volatility, drawdown) therefore cover only
   2006-2025, a subset of the macro panel. This means the Expansion regime
   (pre-2006 months) may have some months without market validation data.

6. **Regime count**: K=4 is not derived from the data — it is imposed by
   design. The notebook does not test whether K=3 or K=5 would produce
   better separation. The core model's silhouette of 0.252 is modest,
   suggesting that 4 clusters may be slightly too many for just 4 indicators.
