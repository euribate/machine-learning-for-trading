# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Where a trading idea meets the data — cross-asset momentum EDA
#
# **Docker image**: `ml4t`
#
# **Purpose**: Before we freeze a setup or fit a model, we test whether one concrete
# idea — **cross-asset momentum** on the 100-ETF universe — actually leaves a
# footprint in the data. This is the exploratory step that sits between a *story*
# ("recent winners keep winning") and a *strategy*. We use the single most
# informative EDA tool in this domain: **sort by the signal, then look at forward
# returns per bucket.**
#
# ## The idea and its mechanism
#
# The idea: assets that went up over the past year tend to keep outperforming over
# the next month, *across* asset classes (equities, sectors, countries, bonds,
# commodities). Before any statistics, name the mechanism — *who is on the other
# side, and why are they happy to trade with us?* This idea sits in the "behind"
# family: we are betting others are slow or a step behind, not that we are paid to
# bear a risk.
#
# - **SLOW**: macro information diffuses across siloed markets with a lag; capital
#   moves to the trend slowly. That gradual adjustment *is* the return.
# - **WRONG**: investors under-react to news and chase performance, extending trends
#   past fundamentals.
#
# So this is a SLOW/WRONG blend. That also tells us the failure mode in advance:
# momentum reverses sharply at turning points, and — because everyone can read the
# same price history — the edge is the kind that **decays**. Keep that hypothesis in
# mind; the data will have something to say about it.
#
# ## Learning objectives
#
# - Build a monthly cross-sectional panel from the ETF daily data.
# - Construct the classic **12-1 month momentum** signal (skip the most recent month).
# - Run the **quintile conditional-return sort** and read it for *monotonicity,
#   magnitude, and shape*.
# - Stress the result by **era** — the difference between a footprint and an artifact.
# - Leave with a measured verdict that motivates the rest of the chapter: freeze the
#   setup, validate walk-forward, and respect costs and effective sample size.
#
# **Book reference**: Chapter 6 §6.1–§6.2 (from idea to evidence; mapping strategies
# and sources of edge); Chapter 7 (defining a tradable label and the conditional
# feature–label relationship).
#
# **Prerequisites**: `data` package on `PYTHONPATH`; ETF parquet present at
# `ML4T_DATA_PATH/etfs/market/`. Run `python data/etfs/market/download.py` if missing.

# %%
"""Cross-asset momentum EDA — from a story to a measured footprint."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_etfs
from utils.style import COLORS, apply_ml4t_style

apply_ml4t_style("light")

# %% tags=["parameters"]
# Production defaults — Papermill injects overrides for CI
LOOKBACK = 12  # months of past return in the signal
SKIP = 1  # skip the most recent month (the "12-1" convention)
N_QUANTILES = 5  # quintiles
MIN_NAMES = 30  # minimum eligible ETFs in a month to include it

# %% [markdown]
# ## 1. Build the monthly panel
#
# The signal and the outcome both live at a **monthly** cadence: we rank the
# universe once a month and hold for the next month. We take each ETF's last close
# of the month. `close` from `load_etfs()` is split/dividend-adjusted, so simple
# returns are total returns.

# %%
prices = (
    load_etfs().to_pandas().pivot(index="timestamp", columns="symbol", values="close").sort_index()
)
monthly = prices.resample("ME").last()

print(f"Daily panel:   {prices.shape[0]:>4} days x {prices.shape[1]} ETFs")
print(f"Monthly panel: {monthly.shape[0]:>4} months x {monthly.shape[1]} ETFs")
print(f"Span:          {monthly.index.min():%Y-%m} to {monthly.index.max():%Y-%m}")

# %% [markdown]
# ## 2. The signal and the outcome
#
# **Signal — 12-1 momentum**: the cumulative return from 12 months ago to *one month
# ago*. Skipping the most recent month is deliberate: it sidesteps the short-term
# reversal that contaminates a raw 12-month return (that reversal is a *different*
# mechanism — liquidity provision — with the opposite sign).
#
# **Outcome — forward 1-month return**: what we actually earn holding the next month.
# Note the two clocks are already distinct: a 12-month **lookback** feeding a 1-month
# **holding period**. Keeping them separate is a core discipline of the strategy
# research framework (§6.3–§6.5).

# %%
momentum = monthly.shift(SKIP) / monthly.shift(LOOKBACK) - 1
forward_ret = monthly.shift(-1) / monthly - 1

# %% [markdown]
# ## 3. The simplest test: sort into quintiles, look at forward returns
#
# Each month we rank every eligible ETF by its momentum signal and split them into
# five equal buckets (quintile 1 = worst past year, quintile 5 = best), form an
# **equal-weight portfolio** of each bucket, and record its return over the *next*
# month. If the mechanism has a footprint, the buckets should line up: higher past
# momentum → higher forward return.
#
# We report each bucket two ways:
#
# - **Gross** — its raw forward return.
# - **Net of the market** — the bucket minus the equal-weight universe that month
#   (its *relative* performance). This strips out beta — the return you get just for
#   being invested — and leaves the momentum *tilt*. Note this is net of the
#   **benchmark**, not net of **trading costs**: we have not subtracted a basis point
#   of cost anywhere in this notebook.
#
# We are reading three things:
#
# - **Monotonicity** — does the relationship order correctly across *all* buckets,
#   not just top-vs-bottom?
# - **Magnitude** — is the top-minus-bottom spread economically material?
# - **Shape** — is the effect smooth (premium-like) or concentrated in the tails
#   (event-like)?

# %%
records = []
bench = {}
for date in momentum.index:
    sig = momentum.loc[date].dropna()
    fwd = forward_ret.loc[date].dropna()
    common = sig.index.intersection(fwd.index)
    if len(common) < MIN_NAMES:
        continue
    sig, fwd = sig[common], fwd[common]
    # rank-then-qcut so ties never collapse a bucket
    quintile = pd.qcut(sig.rank(method="first"), N_QUANTILES, labels=range(1, N_QUANTILES + 1))
    bench[date] = fwd.mean()  # equal-weight universe = the benchmark this month
    for etf in common:
        records.append((date, etf, sig[etf], fwd[etf], int(quintile[etf])))

panel = pd.DataFrame(records, columns=["date", "etf", "momentum", "fwd_ret", "quintile"])
benchmark = pd.Series(bench, name="bench").sort_index()  # monthly equal-weight universe return
print(f"Cross-sectional observations: {len(panel):,} over {panel.date.nunique()} months")


def pct_monthly(r: pd.Series) -> float:
    """Mean forward 1-month return, in percent."""
    return r.mean() * 100


def bps_monthly(r: pd.Series) -> float:
    """Mean forward 1-month return, in basis points."""
    return r.mean() * 1e4


# Monthly equal-weight bucket portfolios, then the same buckets NET of the benchmark.
# We report at the native monthly horizon: the sort ranks once a month and holds for
# one month, so a monthly return is the honest unit. Annualizing a single-month
# cross-sectional sort by ^12 would assume the edge recurs every month for a year —
# which the by-era split below shows it does not.
bucket = panel.groupby(["date", "quintile"])["fwd_ret"].mean().unstack("quintile")
bucket_net = bucket.sub(benchmark, axis=0)
qs = list(range(1, N_QUANTILES + 1))
gross = pd.Series({q: pct_monthly(bucket[q]) for q in qs})  # %/month
net_bps = pd.Series({q: bps_monthly(bucket_net[q]) for q in qs})  # bps/month

print(f"\nBenchmark (equal-weight universe): {pct_monthly(benchmark):.2f}% / month")
print("Forward 1-month return by momentum quintile:")
print(pd.DataFrame({"gross_%/mo": gross.round(3), "net_bps/mo": net_bps.round(1)}))

# The top-minus-bottom spread is identical gross or net — the benchmark cancels.
spread_bps = bps_monthly(bucket[N_QUANTILES]) - bps_monthly(bucket[1])
long_short = (bucket[N_QUANTILES] - bucket[1]).dropna()  # tradable Q5-Q1, dollar-neutral
ls_t = long_short.mean() / long_short.std() * np.sqrt(len(long_short))
print(f"\nTop-minus-bottom spread: {spread_bps:.0f} bps / month (same gross or net)")
print(
    f"Tradable long-short (Q5-Q1): {bps_monthly(long_short):.0f} bps / month, "
    f"t = {ls_t:.2f} (n = {len(long_short)}) — before costs"
)

# %%
fig, (axg, axn) = plt.subplots(1, 2, figsize=(11, 4.6))
labels = [str(q) for q in qs]

bg = axg.bar(labels, gross.values, color=COLORS["slate"], width=0.68)
bg[-1].set_color(COLORS["amber"])
axg.axhline(pct_monthly(benchmark), ls="--", lw=1, color="grey")
axg.set_title("Raw (gross): every bucket rides the market up", loc="left")
axg.set_xlabel("Momentum quintile")
axg.set_ylabel("Forward 1-month return, %")
for x, q in zip(labels, qs):
    v = gross[q]
    axg.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
axg.margins(y=0.16)

bn = axn.bar(labels, net_bps.values, color=COLORS["slate"], width=0.68)
bn[-1].set_color(COLORS["amber"])
axn.axhline(0, ls="--", lw=1, color="grey")
axn.set_title("Net of the market: the momentum tilt", loc="left")
axn.set_xlabel("Momentum quintile  (1 = worst  →  5 = best)")
axn.set_ylabel("Net of market, bps / month")
for x, q in zip(labels, qs):
    v = net_bps[q]
    axn.text(
        x,
        v + (0.8 if v >= 0 else -0.8),
        f"{v:.0f}",
        ha="center",
        va="bottom" if v >= 0 else "top",
        fontsize=10,
    )
axn.margins(y=0.20)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Read it.** In raw (gross) terms every bucket earns 0.5–0.8% a month — but that is
# mostly beta: they all sit around the dashed benchmark line, because the whole
# universe drifted up. Strip the market out (right panel) and what is left is the
# *tilt*: the bottom buckets are slightly negative, the top two slightly positive,
# ordering roughly with past momentum. The spread from worst to best is the same
# either way (~26 bps a month) — the benchmark cancels — and it is **modest**. The
# tradable long-short is thin (t ≈ 0.9, before costs). On the full sample this looks
# like a small footprint; a single average can still hide a lot, so we stress it by era.

# %% [markdown]
# ## 4. Is it lumpy but present, or an artifact? Split by era
#
# A real mechanism is usually *lumpy but present* — it survives being cut by time,
# even if it isn't equally strong everywhere. An effect that lives entirely in one
# period is telling you what it actually is. We split the sample in half and rerun
# the sort. Recall the mechanism's own warning: information-based edges **decay** as
# everyone learns to read the same price history.

# %%
era_of = pd.Series(
    np.where(bucket_net.index < pd.Timestamp("2016-01-01"), "2006–2015", "2016–2025"),
    index=bucket_net.index,
)
rows = []
for era in ["2006–2015", "2016–2025"]:
    m = era_of == era
    q1, q5 = bps_monthly(bucket_net.loc[m, 1]), bps_monthly(bucket_net.loc[m, N_QUANTILES])
    ls_e = (bucket.loc[m, N_QUANTILES] - bucket.loc[m, 1]).dropna()
    rows.append(
        {
            "era": era,
            "Q1_net_bps": round(q1),
            "Q5_net_bps": round(q5),
            "spread_bps": round(q5 - q1),
            "ls_t": round(ls_e.mean() / ls_e.std() * np.sqrt(len(ls_e)), 2),
            "months": int(m.sum()),
        }
    )
era_tbl = pd.DataFrame(rows).set_index("era")
print(era_tbl)

# %%
fig, ax = plt.subplots(figsize=(9, 5))
eras = era_tbl.index.tolist()
x = np.arange(len(eras))
w = 0.38
ax.bar(x - w / 2, era_tbl["Q1_net_bps"], w, label="Worst quintile (Q1)", color=COLORS["slate"])
ax.bar(x + w / 2, era_tbl["Q5_net_bps"], w, label="Best quintile (Q5)", color=COLORS["amber"])
for xi, (q1, q5, sp) in enumerate(
    zip(era_tbl["Q1_net_bps"], era_tbl["Q5_net_bps"], era_tbl["spread_bps"])
):
    ax.text(
        xi - w / 2, q1, f"{q1:.0f}", ha="center", va="bottom" if q1 >= 0 else "top", fontsize=10
    )
    ax.text(
        xi + w / 2, q5, f"{q5:.0f}", ha="center", va="bottom" if q5 >= 0 else "top", fontsize=10
    )
    ax.text(xi, max(q1, q5) + 3, f"spread {sp:.0f} bps", ha="center", va="bottom", fontsize=11)
ax.axhline(0, lw=0.8, color="grey")
ax.set_xticks(x)
ax.set_xticklabels(eras)
ax.set_ylabel("Forward 1-month return, net of market, bps / month")
ax.set_title("Net of the market: the spread collapses across eras", loc="left")
ax.legend(frameon=False, loc="lower right")
ax.margins(y=0.2)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Read it.** Net of the market, the tilt is clear in the first decade (a wide
# Q5-over-Q1 gap) and much smaller in the second — the spread falls to roughly a fifth
# of its first-decade level (about 45 bps to about 9 bps a month).
# That is consistent with an information-based edge decaying as it becomes widely
# known. Two cautions keep this at "suggestive," not "proven": the tradable
# long-short is not statistically significant in either era (see `ls_t`), and a
# two-bucket split is itself a coarse cut — the exercises push you to move the
# boundary and watch how stable the fade really is.

# %% [markdown]
# ## 5. One feasibility check: how fast does the signal turn over?
#
# Costs are decided by turnover, and turnover is governed by how much the ranking
# changes month to month. We measure the month-over-month rank autocorrelation of
# the signal: high autocorrelation means today's winners are mostly last month's
# winners, so we trade little.

# %%
autocorrs = []
dates = momentum.index
for prev, curr in zip(dates[:-1], dates[1:]):
    a, b = momentum.loc[prev].dropna(), momentum.loc[curr].dropna()
    common = a.index.intersection(b.index)
    if len(common) < MIN_NAMES:
        continue
    autocorrs.append(a[common].corr(b[common], method="spearman"))
print(f"Mean month-over-month rank autocorrelation: {np.mean(autocorrs):.3f}")
print("(High → the ranking is persistent → turnover and costs are modest.)")

# %% [markdown]
# ## 6. Verdict — and what comes next
#
# What the EDA established, and what it did **not**:
#
# - Net of the market, the mechanism leaves a **modest, top-concentrated** tilt —
#   enough to be worth engineering, not enough to trade on its own. The tradable
#   long-short is thin and **not statistically significant** (t ≈ 0.9), and all of
#   this is **before costs**.
# - It is **regime-dependent**: the spread collapses from about 45 bps a month in
#   2006–2015 to about 9 bps in 2016–2025. Evaluation must respect *when* the edge lived, which is why we validate
#   **walk-forward**, not on a shuffled or full-sample average.
# - The signal is **persistent** (low turnover), so costs are unlikely to erase it
#   outright — but again, we have not yet subtracted a single basis point.
#
# Nothing here is a backtest. It is visual confirmation that there is *something* to
# model, plus a set of cautions. The rest of the chapter turns this into a disciplined
# evaluation: **freeze the setup** (§6.3), **validate it walk-forward without leakage**
# (§6.5), and keep the search **countable and auditable** (§6.7). Chapter 7 then
# defines a **tradable label** and accounts for **overlap and effective sample size**
# before trusting any significance.
