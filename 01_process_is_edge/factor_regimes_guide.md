# Factor Regimes Notebook - Operative Guide

**Notebook**: `01_process_is_edge/factor_regimes.ipynb`
**Book reference**: Chapter 1, Section 1.4 "Market Regimes: Change Is the Constant"
**Last verified**: 2026-07-02 (AQR data through Feb 2026)

---

## Table of Contents

1. [Purpose and Context](#1-purpose-and-context)
2. [Models Used](#2-models-used)
3. [Step-by-Step Code Walkthrough](#3-step-by-step-code-walkthrough)
4. [Comment Alignment Audit](#4-comment-alignment-audit)
5. [How to Interpret the Results](#5-how-to-interpret-the-results)
6. [Key Caveats](#6-key-caveats)

---

## 1. Purpose and Context

This notebook demonstrates **unsupervised regime detection** on financial factor
returns. The goal is to partition a century of monthly return data into distinct
market regimes (e.g., Risk-On vs Risk-Off) using Gaussian Mixture Models (GMM),
then characterize how factor strategies behave differently across those regimes.

### Why this matters for trading

Markets are not stationary: correlations, volatilities, and return distributions
shift over time. A momentum strategy that thrives in calm, trending markets can
collapse during sharp reversals. Regime detection aims to identify these structural
shifts. Even though this notebook is *descriptive* (ex-post, full-sample fitting),
it establishes the conceptual framework that the book later makes predictive in
Chapters 6-20.

### Inspiration

The approach is inspired by Two Sigma's 2021 paper "A Machine Learning Approach to
Regime Modeling", which uses GMM on an 18-factor "Factor Lens" to identify four
market regimes. This notebook adapts the idea to AQR's longer-history dataset.

### Scope: descriptive, NOT predictive

The GMM and StandardScaler are fitted on the **entire** factor-return history
(1927-2026) and labels are assigned back over that same history. This is an
*ex-post* characterization. Using these labels as trading features would constitute
**look-ahead bias**. The notebook states this clearly. Walk-forward and
embargoed cross-validation for predictive regime models are covered from Chapter 6
onward.

---

## 2. Models Used

### 2.1 Gaussian Mixture Model (GMM)

**What it is**: A probabilistic generative model that assumes the data was generated
by a mixture of K multivariate Gaussian distributions. Each Gaussian component has
its own mean vector and covariance matrix. The model estimates:
- The mean and covariance of each component (cluster)
- The mixing weights (prior probability of each component)
- Soft assignments: each data point gets a probability of belonging to each cluster

**Why GMM over K-Means for financial data**:

| Property | GMM | K-Means |
|----------|-----|---------|
| Cluster shape | Ellipsoidal (full covariance) | Spherical only |
| Assignment | Soft (probabilities) | Hard (nearest centroid) |
| Handles correlated features | Yes (via covariance matrix) | No |
| Model selection criteria | BIC, AIC (principled) | Silhouette only (heuristic) |

Financial factor returns exhibit strong cross-correlations (e.g., equity market and
carry tend to move together). GMM with `covariance_type="full"` captures these
correlations. K-Means assumes spherical clusters with equal variance across
dimensions, which is a poor assumption for correlated factor returns.

**Key hyperparameters in this notebook**:
- `n_components`: swept from 2 to 6
- `covariance_type="full"`: each component has its own unrestricted covariance matrix
- `n_init=10`: the model is fitted 10 times with different random initializations;
  the best result (highest log-likelihood) is kept. This guards against local optima
  in the EM algorithm.
- `reg_covar=1e-6`: small regularization added to the diagonal of covariance matrices
  to prevent singular matrices when a component collapses onto a small set of points.
- `random_state=42`: reproducibility

### 2.2 K-Means (comparison baseline)

Fitted alongside GMM to show that both methods agree on the broad structure, but
K-Means produces lower silhouette scores because it cannot capture the ellipsoidal
shape of the clusters.

### 2.3 Model Selection Criteria

**BIC (Bayesian Information Criterion)**:
`BIC = -2 * log_likelihood + k * ln(n)`
where k = number of free parameters, n = number of data points.
BIC penalizes model complexity heavily. **Lower is better**. It is the standard
choice when the goal is interpretability and generalization, not maximum likelihood.

**AIC (Akaike Information Criterion)**:
`AIC = -2 * log_likelihood + 2 * k`
**Lower is better**. AIC penalizes complexity less than BIC (the penalty term is
`2k` vs `k * ln(n)`, and `ln(n) > 2` for any dataset with more than 7 observations).
As a result, AIC tends to favor more complex models. In this notebook, AIC
decreases monotonically from K=2 (25,837) to K=6 (25,148), so based on AIC alone
you would choose K=6 — the most complex model in the grid.

**Why report AIC if BIC is preferred?** AIC and BIC answer different questions.
BIC approximates `-2 * ln(marginal likelihood)` — the negative log of the
probability that the model generated the data (Bayesian model evidence). Because
of the negative sign, lower BIC means higher marginal likelihood, i.e. the model
is more probable. BIC is *consistent*: given enough data, it will select the true
number of components if the data truly comes from a mixture. AIC instead minimizes
the expected out-of-sample prediction error (Kullback-Leibler divergence) and is
*efficient*: it selects the model that best predicts new observations, even if the
"true" model is not in the candidate set. In practice:

- Use **BIC** when the goal is to find the simplest adequate model (regime
  identification, interpretability, narrative). This is the case in this notebook.
- Use **AIC** when the goal is out-of-sample density estimation or forecasting, where
  a slightly more complex model may capture structure that BIC would prune away.
- Reporting both exposes the tension between parsimony and fit, which is itself
  informative: when BIC and AIC agree, the choice is unambiguous; when they disagree
  (as here), it signals that additional clusters improve fit but not enough to justify
  their complexity under the stricter BIC criterion.

**Silhouette Score**:
Measures how similar each point is to its own cluster vs. the nearest other cluster.
Range: [-1, 1]. Values near 1 mean well-separated clusters; near 0 means overlapping
clusters; negative means points are assigned to the wrong cluster. It is a
distance-based heuristic, not model-based, so it is less appropriate for mixture
models than BIC/AIC but still useful as a sanity check.

### 2.4 StandardScaler

Applied before clustering so that all 9 factors contribute equally. Without scaling,
factors with larger variance (e.g., equity market returns ~15% annual vol) would
dominate the distance calculations over factors with smaller variance (e.g., carry
~3% annual vol). StandardScaler removes the mean and scales to unit variance.

---

## 3. Step-by-Step Code Walkthrough

### Step 1: Imports and Configuration

```python
SEED = 42
OUTPUT_DIR = get_output_dir(1, "factor_regimes")
set_global_seeds(SEED)
```

**Why**: Sets the random seed for reproducibility. `set_global_seeds()` seeds
numpy, Python's random module, and any other relevant generators. The output
directory follows the project convention (`01_process_is_edge/output/factor_regimes/`).

### Step 2: Helper - GmmFitResult dataclass

```python
@dataclass(frozen=True)
class GmmFitResult:
    model: GaussianMixture
    labels: np.ndarray
    probabilities: np.ndarray
    bic: float
    aic: float
    silhouette: float
```

**Why**: Bundles the fitted model with its diagnostics in a single immutable
container. This avoids passing around separate variables and makes the grid search
results clean to inspect.

### Step 3: Grid Search Function

```python
def fit_gmm_grid(x, n_components_list, random_state=SEED):
```

**What it does**: For each candidate K in [2, 3, 4, 5, 6]:
1. Fits a GMM with K components
2. Predicts hard labels and soft probabilities
3. Computes BIC, AIC, and silhouette score
4. Returns a dictionary mapping K -> GmmFitResult

**Why a grid search**: There is no objectively "correct" number of regimes.
The grid search lets us evaluate multiple candidates and choose based on
statistical criteria rather than guessing.

### Step 4: Load AQR Century of Factor Premia

```python
aqr_raw_pl = AQRFactorProvider().fetch("century_premia")
```

**Output**: 1,196 months (Jul 1926 - Feb 2026), 44 factor columns.

**Why this dataset**: AQR's dataset provides nearly 100 years of monthly
factor returns across multiple asset classes (equities, fixed income,
commodities). The long history spans multiple market cycles, including
the Great Depression, WWII, stagflation of the 1970s, the dot-com bubble,
the GFC, and COVID. This makes it ideal for regime detection because the
model can see diverse market environments.

### Step 5: Factor Selection

Nine factors are selected from the 44 available:

| Factor | Why included |
|--------|-------------|
| All asset classes Value | Core style factor - buy cheap, sell expensive |
| All asset classes Momentum | Core style factor - buy winners, sell losers |
| All asset classes Carry | Core style factor - buy high yield, sell low yield |
| All asset classes Defensive | Core style factor - buy low-risk, sell high-risk |
| US Stock Selection Value | US-specific value to capture domestic dynamics |
| US Stock Selection Momentum | US-specific momentum |
| Equity indices Market | Broad equity market return (the "beta" factor) |
| Fixed income Market | Bond market return |
| Commodities Market | Commodity market return |

**Why these 9**: They span the main asset classes (equity, fixed income,
commodities) and the main style factors (value, momentum, carry, defensive).
This gives the GMM a multi-dimensional view of the market state. Missing
data is minimal (only US Stock Selection Momentum has 0.5% missing).

### Step 6: Data Preparation

```python
factors_pl = aqr_raw_pl.select(...).sort("timestamp").fill_null(strategy="forward").drop_nulls()
scaler = StandardScaler()
factors_scaled = scaler.fit_transform(factors_df)
```

**Processing**:
1. Select the 9 factors + timestamp
2. Sort chronologically
3. Forward-fill nulls (6 missing values in US Stock Selection Momentum)
4. Drop remaining nulls (rows at the start with no prior value to forward-fill)
5. Result: **1,190 months** (1927-2026)
6. StandardScaler: zero mean, unit variance for each factor

**Why forward-fill**: The 6 missing values are early in the series. Forward-fill
is the standard approach for sporadic gaps in time-series: it assumes the last
known value persists until the next observation. It avoids introducing bias from
interpolation or dropping entire rows.

### Step 7: GMM Grid Search + K-Means Comparison

```python
gmm_grid = fit_gmm_grid(factors_scaled, [2, 3, 4, 5, 6])
# + K-Means for each K
```

**Actual results** (verified 2026-07-02):

| K | BIC | AIC | Silhouette (GMM) | Silhouette (K-Means) |
|---|-----|-----|------------------|---------------------|
| 2 | 26,391 | 25,837 | 0.288 | 0.234 |
| 3 | 26,366 | 25,532 | 0.126 | 0.117 |
| 4 | 26,417 | 25,304 | 0.123 | 0.125 |
| 5 | 26,634 | 25,241 | 0.110 | 0.091 |
| 6 | 26,820 | 25,148 | 0.036 | 0.078 |

**Interpretation**:
- BIC is minimized at K=3 (26,366), with K=2 very close (26,391 — a difference of
  just 25 points, well within noise for BIC comparisons)
- Silhouette is maximized at K=2 (0.288 GMM, 0.234 K-Means) and drops sharply at K=3
- AIC keeps decreasing through K=6 (as expected — AIC has weaker complexity penalty)
- GMM silhouette >= K-Means silhouette at every K except K=4 (0.123 vs 0.125),
  confirming that GMM's ellipsoidal clusters better fit the correlated factor data
- **K=2 is chosen**: BIC is near-minimal and silhouette is clearly highest, favoring
  the simpler, more interpretable two-regime split

### Step 8: Model Selection Visualization

Three bar charts (BIC, AIC, Silhouette) with the best K highlighted in gold.
The y-axes are zoomed to emphasize differences. The suptitle states the
conclusion: "BIC and Silhouette Agree on K=2; AIC Prefers K=6".

### Step 9: Regime Timeline Swim Lanes

For each K (2 through 6), a swim-lane heatmap shows which regime is active at
each month. This visualization reveals:
- K=2: clean, interpretable split
- K=3-4: increasingly fragmented
- K=5-6: noisy, with some regimes appearing only briefly

### Step 10: Two-Regime Labeling

```python
regime_equity_returns = factors_df["Equity indices Market"].groupby(labels_2).mean()
good_regime = regime_equity_returns.idxmax()
bad_regime = 1 - good_regime
```

**Why**: GMM labels are arbitrary (0 or 1). The code assigns "Risk-On" to the
regime with higher average equity returns and "Risk-Off" to the other. This is
purely for interpretability — it does not change the model.

**Result**: Risk-On = 927 months (77.9%), Risk-Off = 263 months (22.1%).

### Step 11: Two-Regime Timeline with Historical Events

A two-panel figure:
- **Top**: Swim lanes for Risk-On / Risk-Off, with vertical lines at major events
  (1929 Great Crash, 1937 Recession, 1973 Oil Crisis, 1987 Black Monday,
  2000 Dot-com, 2008 GFC, 2020 COVID)
- **Bottom**: Cumulative equity return (log scale), colored by regime

**How to read it**: In the top panel, the two swim-lane rows are "Risk-Off"
(bottom) and "Risk-On" (top). A dark fill in a row means that regime is active
at that month. Red vertical lines and labels mark major historical events
(1929 Great Crash, 1937 Recession, 1973 Oil Crisis, 1987 Black Monday,
2000 Dot-com, 2008 GFC, 2020 COVID). To check whether a crisis coincides with
Risk-Off, look at whether the Risk-Off row (bottom) is dark (filled) where the
red line falls.

Note that the GMM assigns regimes based on factor returns, not event narratives.
A crisis may start with the model still in Risk-On and only flip to Risk-Off
once factor dislocations materialize (often with a one- or two-month lag). Short
or localized shocks (e.g., Black Monday — a single-day crash) may not generate
enough sustained factor stress to flip the monthly regime at all. The alignment
between events and Risk-Off is a general pattern, not a one-to-one match.

### Step 12: Volatility Analysis

12-month rolling volatility (annualized) is computed and split by regime.

**Actual results** (verified):

| Regime | Mean Vol | Median Vol | Max Vol |
|--------|----------|------------|---------|
| Risk-On | 9.8% | 8.9% | — |
| Risk-Off | 12.6% | 11.0% | — |

Rolling-vol ratio: **1.29x** (Risk-Off is 29% more volatile on average, using
the smoothed rolling measure).

The figure includes the regime swim-lane panel on top and rolling volatility
time series below, with dashed horizontal lines at each regime's mean.

**Artifact persistence**: The regime labels, rolling volatility, and regime
means are saved to `output/factor_regimes/figure_1_5/inputs.npz` for use by
the book's publication-quality figure generation script.

### Step 13: Factor Returns by Regime

Mean monthly returns are annualized (x12) and displayed as a grouped bar chart.

**Actual results** (verified 2026-07-02):

| Factor | Risk-On (%) | Risk-Off (%) |
|--------|-------------|--------------|
| Value | +1.7 | +5.1 |
| Momentum | +4.5 | +0.1 |
| Carry | +3.5 | -1.3 |
| Defensive | +4.2 | -1.0 |
| US Value | -0.4 | +19.8 |
| US Momentum | +10.2 | -0.2 |
| Equity | +10.0 | +0.7 |
| Bonds | +1.1 | +2.7 |
| Commodities | +4.4 | +7.4 |

### Step 14: Regime Statistics

Comprehensive statistics for each regime (equity market returns only):

| Metric | Risk-On | Risk-Off |
|--------|---------|----------|
| Months | 927 | 263 |
| % of Time | 77.9% | 22.1% |
| Ann. Return | +10.0% | +0.7% |
| Ann. Volatility | 8.6% | 19.4% |
| Sharpe Ratio | 1.17 | 0.04 |
| Max Drawdown | -24.3% | -83.5% |

Direct volatility ratio: **2.26x** (much higher than the rolling 1.29x because
the direct measure captures the full variance of all months within each regime,
including extreme months that the 12-month rolling window smooths out).

### Step 15: Duration Analysis

**Actual results** (verified):

| Metric | Value |
|--------|-------|
| Total transitions | 275 |
| Avg months between transitions | 4.3 |
| Risk-On avg duration | 6.7 months |
| Risk-On max duration | 110 months |
| Risk-On episodes | 138 |
| Risk-Off avg duration | 1.9 months |
| Risk-Off max duration | 14 months |
| Risk-Off episodes | 138 |

---

## 4. Comment Alignment Audit

The notebook's inline markdown interpretations contain specific numbers that were
written against an earlier snapshot of the AQR data (likely ending 2024). With the
dataset now extending through Feb 2026 (1,196 rows), several values have shifted.

### Misaligned Comments

| Location | Comment claims | Actual value | Severity |
|----------|---------------|--------------|----------|
| **Model Selection** cell | "BIC is minimised at K=2 (26,170)" | BIC is minimized at K=3 (26,366); K=2 is 26,391 | **Incorrect** |
| **Model Selection** cell | Silhouette at K=2 = 0.27 | 0.288 | Minor |
| **Model Selection** cell | AIC at K=6 = 24,930 | 25,147.7 | Minor |
| **Model Selection** cell | "silhouette turns negative for K>=4" | All positive (0.123, 0.110, 0.036) | **Incorrect** |
| **Model Selection** cell | "1927-2024 panel" | 1927-2026 panel | Minor |
| **Factor Returns** cell | Value: +5.3% vs +1.6% | +5.1% vs +1.7% | Minor |
| **Factor Returns** cell | Momentum: +0.4% Risk-Off | +0.1% | Minor |
| **Factor Returns** cell | Carry: -0.6% Risk-Off | -1.3% | Moderate |
| **Factor Returns** cell | Defensive: -0.5% Risk-Off | -1.0% | Moderate |
| **Factor Returns** cell | Bonds: +4.2% vs +0.8% | +2.7% vs +1.1% | Moderate |
| **Regime Statistics** cell | Sharpe 1.11 vs 0.12 | 1.17 vs 0.04 | Moderate |
| **Regime Statistics** cell | Max DD -76.9% | -83.5% | Moderate |
| **Duration Analysis** cell | "267 transitions" | 275 | Minor |
| **Duration Analysis** cell | "98 years" | 99 years (1927-2026) | Minor |
| **Key Takeaways** cell | Repeats all above misaligned values | — | See above |

### Qualitative Assessment

Despite the numeric shifts, **all qualitative conclusions remain correct**:

- BIC is actually minimized at K=3 (26,366 vs 26,391 at K=2), but silhouette
  clearly favors K=2 (0.288 vs 0.126). The choice of K=2 is justified by
  silhouette and parsimony, not by BIC alone
- Value is still countercyclical (correct)
- Momentum is still procyclical (correct)
- Carry and Defensive still turn negative in Risk-Off (correct, and more so)
- Bonds still outperform in Risk-Off (correct, though less dramatically)
- Risk-Off Sharpe is still near zero (correct, and even worse: 0.04 vs claimed 0.12)
- Volatility ratio is still ~2.2x direct and ~1.3x rolling (correct)
- Regime signals are still noisy with ~4 month average transitions (correct)

### Factual Errors

1. **"BIC is minimised at K=2"** — Wrong. BIC is minimized at K=3 (26,366 vs
   26,391 at K=2). The difference is small (25 points), but the claim is factually
   incorrect. K=2 is still a defensible choice based on silhouette (0.288 vs 0.126)
   and parsimony, but the justification should not invoke BIC minimality.

2. **"silhouette scores turn negative for K>=4"** — Wrong. All silhouette scores
   remain positive (0.123 at K=4, 0.110 at K=5, 0.036 at K=6). They are low and
   declining, indicating increasingly poor separation, but they do not cross zero.
   The sentence should be corrected to something like: "Silhouette scores drop
   sharply for K>=3 and approach zero at K=6, indicating increasingly overlapping
   clusters."

---

## 5. How to Interpret the Results

### 5.1 What the Two-Regime Split Means

The GMM identifies two distinct states of the factor world:

**Risk-On (78% of time)**: Equity markets deliver strong returns (+10.0%
annualized), volatility is moderate (8.6%), and most factor strategies work well.
Momentum (+4.5%) and Defensive (+4.2%) are the strongest style factors. This is
the "normal" state of markets.

**Risk-Off (22% of time)**: Equity returns collapse to near zero (+0.7%),
volatility doubles (19.4%), and factor correlations shift. Value becomes the
dominant factor (+5.1%), consistent with the academic view that the value premium
compensates for distress risk — it pays off precisely when holding equities is
most painful. Momentum breaks down (0.1%), as sharp market reversals punish
trend-following strategies. Carry and Defensive, despite their "safe" names,
lose money (-1.3% and -1.0%).

### 5.2 Regime Duration

Risk-Off episodes are short and sharp: average 1.9 months, max 14 months. They
represent acute stress (crashes, crises) rather than prolonged bear markets.
The model transitions every 4.3 months on average (275 transitions over 99 years),
making it too noisy for direct tactical use without additional filtering.

### 5.3 Two Volatility Measures

The notebook reports two volatility ratios:
- **Rolling (1.29x)**: Based on 12-month rolling standard deviation. This measure
  smooths over extreme months and reflects the "local" volatility environment.
- **Direct (2.26x)**: Based on the standard deviation of all returns within each
  regime. This captures the full variance including extreme months.

The direct measure is more appropriate for risk management because it reflects
what a portfolio actually experiences. The rolling measure understates tail risk.

### 5.4 Portfolio Construction Implications

The results suggest:
- **Value and Bonds** provide genuine diversification during stress
- **Carry and Defensive** do not provide stress diversification despite their names
- **Momentum** is the worst performer during Risk-Off
- A portfolio that reduces momentum/carry exposure and increases value/bond
  exposure during Risk-Off would (ex-post) improve risk-adjusted returns

However, this requires knowing the regime in real time — which this notebook
explicitly does not do. Predictive regime models are built in later chapters.

---

## 6. Key Caveats

1. **Look-ahead bias**: The entire analysis is ex-post. Labels are fitted on
   the full sample. Do not use these labels as trading features.

2. **Stationarity assumption**: GMM assumes the data-generating process is a
   fixed mixture of Gaussians. In reality, market regimes evolve over time.
   The distributions in the 1930s are not the same as in the 2020s.

3. **Feature engineering matters**: The 9 factors chosen determine the regime
   structure. Different factor sets produce different regime maps. There is
   no single "true" regime decomposition.

4. **GMM is sensitive to initialization**: The `n_init=10` parameter mitigates
   this, but results can still vary slightly across runs if the seed changes.

5. **Short Risk-Off episodes**: Average duration of 1.9 months means many
   Risk-Off periods are just 1-2 months. A strategy acting on these signals
   would trade frequently, eroding returns through transaction costs.

6. **The data updates**: AQR periodically updates the Century of Factor Premia
   dataset. New data changes the GMM fit and all derived statistics. The
   specific numbers in the notebook comments were calibrated to an earlier
   version (likely ending 2024) and drift as new months are added.
