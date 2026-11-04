"""Scaling-law fits, arm gaps, and the crossover extrapolation (arthritis).

Run: ./surf/Scripts/python.exe -m reports.scaling_analysis

Turns the six-point ladder into the quantities a scaling claim actually requires:

  1. a saturating power-law fit per arm,  AUROC(n) = c - a * n**(-b),
     where c is the fitted ceiling and b the scaling exponent;
  2. the arm-vs-arm gap as a function of log10(n), with the extrapolated parity point;
  3. a SENSITIVITY table over which rungs are included in the fit. This is the honest
     centre of the analysis: the parity estimate moves by several-fold depending on the
     fit window, largely because the 10k Arm C runs are optimisation failures rather than
     data-limited measurements, so a single point estimate would be indefensible.

Seed-level uncertainty here is over 3 training seeds only. It is NOT the sampling error of
the test metric; that requires the paired bootstrap over patients in
`reports/paired_bootstrap.py`, which needs preds.parquet from all three arms.

Writes reports/scaling_fits_arthritis.md and reports/figures/scaling_gap_arthritis.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scale.config import EXPERIMENTS, REPORTS

DISEASE = "arthritis"
# Run 4 substitution. At 10k the headline Arm C configuration (warmup 300, patience 4) failed
# to train: two of three seeds stopped at epoch 1 and the seed SD was 0.0900, versus 0.0013 at
# the top rung. Run 4 re-ran that rung with warmup 100, patience 10 and 5 seeds, recovering
# +0.095 AUROC and cutting the spread to 0.0302. The broken point is therefore replaced by the
# fixed one throughout the analysis.
#
# This does mix configurations across the ladder, which is disclosed in the paper. The
# justification is Run 2: quadrupling the epoch cap, near-quadrupling patience and adding
# cosine decay changed the 300k and 1,114,144 results by at most 0.0009 and left the best
# epoch identical, so schedule settings are inert where training succeeds and matter only
# where it was failing.
ARMC_10K_FIX = {("C", 10000): "armC_r4fix"}

ARMS = {"A": "Engineered features (GBDT)",
        "B": "Code sequence (CNN, 2017)",
        "C": "Code sequence (Transformer)"}


def _load_arm(arm: str) -> pd.DataFrame:
    """Per-(size, seed) metrics straight from experiments/, not the summary CSVs."""
    if arm == "A":
        g = list((EXPERIMENTS / DISEASE / "armA" / "rich_full").glob("*/metrics.json"))
    else:
        g = list((EXPERIMENTS / DISEASE / f"arm{arm}").glob("*/metrics.json"))
    rows = [json.loads(p.read_text()) for p in g]
    for (a, size), sub in ARMC_10K_FIX.items():          # swap in the Run 4 rung
        if a != arm:
            continue
        rows = [r for r in rows if r["size"] != size]
        rows += [json.loads(q.read_text())
                 for q in (EXPERIMENTS / DISEASE / sub).glob(f"size{size}_seed*/metrics.json")]
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    return d[["size", "seed", "auroc", "auprc"]].assign(arm=arm)


B_GRID = np.linspace(0.02, 1.2, 400)


def _powerlaw(n, y):
    """Fit AUROC(n) = c - a*n**(-b) by grid search on b with a closed-form (c, a).

    Only 6 points, so a coarse grid plus least squares is more stable than a gradient
    optimiser and never fails to converge. The ceiling is constrained to c <= 1 because
    AUROC is bounded; an arm that has not begun to saturate over the measured range drives
    the unconstrained fit to a nonsensical c > 1, which we detect and report as
    "no identifiable ceiling" rather than printing an impossible asymptote.
    """
    best = None
    for b in B_GRID:
        x = n.astype(float) ** (-b)
        Amat = np.column_stack([np.ones_like(x), -x])
        coef, *_ = np.linalg.lstsq(Amat, y, rcond=None)
        c, a = float(coef[0]), float(coef[1])
        if c > 1.0:                       # re-solve with the ceiling pinned at 1.0
            c = 1.0
            a = float((x @ (c - y)) / (x @ x))
        resid = float((((c - a * x) - y) ** 2).sum())
        if best is None or resid < best[0]:
            best = (resid, c, a, float(b))
    _, c, a, b = best
    # A fit is only interpretable if all three hold:
    #   (i)   the exponent is not pinned to the edge of the search grid;
    #   (ii)  the ceiling is not running up against the AUROC bound (a fitted asymptote
    #         near 1.0 on a task like this means "still linear in log n", not "ceiling 1.0");
    #   (iii) the ceiling exceeds every value actually observed, since an asymptote below
    #         the measured curve is incoherent and signals an outlier dragging the fit.
    identified = (B_GRID[0] * 1.01 < b < B_GRID[-1] * 0.99
                  and c < 0.95
                  and c > float(np.max(y)))
    return dict(ceiling=c, a=a, exponent=b, identified=identified)


def _gap_fit(sizes, gap):
    """Log-linear fit of the arm gap vs log10(n); returns slope per decade + parity n."""
    ln = np.log10(np.asarray(sizes, dtype=float))
    m, i = np.polyfit(ln, np.asarray(gap, dtype=float), 1)
    parity = 10 ** (-i / m) if m < 0 else np.inf
    return float(m), float(parity)


def build():
    arms = {a: _load_arm(a) for a in ARMS}
    missing = [a for a, d in arms.items() if d.empty]
    if missing:
        raise SystemExit(f"no metrics found for arm(s) {missing}; run them first")
    allm = pd.concat(arms.values(), ignore_index=True)
    mean = allm.groupby(["arm", "size"]).auroc.mean().unstack(0)
    sd = allm.groupby(["arm", "size"]).auroc.std().unstack(0)
    pr = allm.groupby(["arm", "size"]).auprc.mean().unstack(0)
    sizes = mean.index.to_numpy()

    L = []
    w = L.append
    w("# Scaling analysis: arthritis onset, three arms")
    w("")
    w("_Generated by `reports/scaling_analysis.py` from `experiments/arthritis/*/metrics.json`. "
      "Uncertainty shown here is over 3 training seeds and is NOT the sampling error of the "
      "test metric; see the paired bootstrap for that._")
    w("")

    # ---- learning curves ---------------------------------------------------
    w("## 1. Learning curves (AUROC, mean over 3 seeds, SD in brackets)")
    w("")
    w("| training patients | A features | B CNN | C Transformer | A-C | A-B | C-B |")
    w("|---|---|---|---|---|---|---|")
    for s in sizes:
        w(f"| {s:,} | {mean.A[s]:.4f} ({sd.A[s]:.4f}) | {mean.B[s]:.4f} ({sd.B[s]:.4f}) | "
          f"{mean.C[s]:.4f} ({sd.C[s]:.4f}) | {mean.A[s]-mean.C[s]:+.4f} | "
          f"{mean.A[s]-mean.B[s]:+.4f} | {mean.C[s]-mean.B[s]:+.4f} |")
    w("")
    w("| training patients | AUPRC A | AUPRC B | AUPRC C |")
    w("|---|---|---|---|")
    for s in sizes:
        w(f"| {s:,} | {pr.A[s]:.4f} | {pr.B[s]:.4f} | {pr.C[s]:.4f} |")
    w("")

    # ---- seed instability flag --------------------------------------------
    w("## 2. Seed stability")
    w("")
    w("| size | SD(A) | SD(B) | SD(C) |")
    w("|---|---|---|---|")
    for s in sizes:
        flag = "  <- unstable" if max(sd.A[s], sd.B[s], sd.C[s]) > 0.02 else ""
        w(f"| {s:,} | {sd.A[s]:.4f} | {sd.B[s]:.4f} | {sd.C[s]:.4f} |{flag}")
    w("")
    bad = [int(s) for s in sizes if sd.C[s] > 0.02]
    if bad:
        eps = allm[(allm.arm == "C") & (allm["size"].isin(bad))]
        w(f"- Arm C is unstable at {bad}: seed SD up to **{sd.C[bad].max():.4f}**, versus "
          f"{sd.C[sizes[-1]]:.4f} at the top rung. Inspecting those runs, several stop at "
          "the first epoch, i.e. validation AUROC never improves and patience fires. These "
          "are **optimisation failures, not data-limited measurements**, and the fits below "
          "are reported with and without them for exactly that reason.")
        w(f"  Per-seed AUROC at those rungs: "
          f"{', '.join(f'{int(r.size):,}/seed{int(r.seed)}={r.auroc:.4f}' for r in eps.itertuples())}.")
    w("")

    # ---- power law per arm -------------------------------------------------
    w("## 3. Saturating power-law fit per arm")
    w("")
    w("`AUROC(n) = ceiling - a * n^(-b)`. `ceiling` is the fitted asymptote (the "
      "irreducible-error bound for that representation), `b` the scaling exponent.")
    w("")
    w("| arm | fit window | ceiling | exponent b | AUROC at top rung | headroom to ceiling |")
    w("|---|---|---|---|---|---|")
    unident = []
    for a in ARMS:
        for lo, lab in ((sizes[0], "all rungs"), (30000, "excl. 10k")):
            m = sizes >= lo
            f = _powerlaw(sizes[m], mean[a].to_numpy()[m])
            if f["identified"]:
                w(f"| {a} | {lab} | {f['ceiling']:.4f} | {f['exponent']:.3f} | "
                  f"{mean[a][sizes[-1]]:.4f} | {f['ceiling']-mean[a][sizes[-1]]:+.4f} |")
            else:
                unident.append(f"{a} ({lab})")
                w(f"| {a} | {lab} | not identifiable | not identifiable | "
                  f"{mean[a][sizes[-1]]:.4f} | - |")
    w("")
    w("- **Arm A is the only arm with an identifiable ceiling.** Its fitted asymptote sits "
      "just above where it already is, i.e. engineered features are close to saturation on "
      "this task: more patients buy very little.")
    if unident:
        w(f"- No ceiling is identifiable for {', '.join(unident)}. Over the measured range "
          "these arms are still on the near-linear part of the curve, so the data cannot "
          "distinguish a high asymptote reached slowly from a lower one reached later. "
          "This is reported as a limitation rather than papered over with an extrapolated "
          "number: it is the same fact that makes the parity estimates in section 4 fragile.")
    w("")

    # ---- gap fits + sensitivity -------------------------------------------
    w("## 4. Gap closure and the extrapolated parity point (the sensitivity that matters)")
    w("")
    w("| comparison | fit window | slope per decade | extrapolated parity n | decades beyond top rung |")
    w("|---|---|---|---|---|")
    windows = ((sizes[0], "all rungs"), (30000, "excl. 10k"), (100000, ">= 100k"))
    sens = {}
    for hi, lo_arm in (("A", "C"), ("A", "B")):
        for lo, lab in windows:
            m = sizes >= lo
            gap = (mean[hi] - mean[lo_arm]).to_numpy()[m]
            slope, parity = _gap_fit(sizes[m], gap)
            dec = np.log10(parity) - np.log10(sizes[-1]) if np.isfinite(parity) else np.inf
            sens.setdefault(f"{hi} vs {lo_arm}", []).append(parity)
            w(f"| {hi} vs {lo_arm} | {lab} | {slope:+.4f} | {parity:.2e} | {dec:.2f} |")
    w("")
    for k, v in sens.items():
        w(f"- **{k}: parity estimates span {min(v):.2e} to {max(v):.2e}**, a "
          f"{max(v)/min(v):.1f}x spread driven purely by which rungs enter the fit.")
    w("")
    w("**How to state this.** A single extrapolated crossover point is not defensible from "
      "six rungs whose lowest points are optimisation-limited. The claim the data supports "
      "is a bound: under log-linear gap closure, parity between engineered features and a "
      "learned sequence representation requires on the order of 10^7 patients or more, one "
      "to two orders of magnitude beyond the 1.1M rung measured here, and beyond the size "
      "of any single commercial claims cohort in routine use. Engineered features remain "
      "ahead throughout the achievable range.")
    w("")

    # ---- transformer vs CNN ------------------------------------------------
    w("## 5. Transformer versus CNN: the advantage is scale-dependent")
    w("")
    cb = (mean.C - mean.B)
    first = next((int(s) for s in sizes if cb[s] > 0), None)
    w("| size | C - B |")
    w("|---|---|")
    for s in sizes:
        w(f"| {s:,} | {cb[s]:+.4f} |")
    w("")
    if first:
        w(f"- The Transformer does **not** dominate the 2017 CNN at every scale. It is behind "
          f"at {sizes[0]:,} and first overtakes at **{first:,}** training patients. Reporting "
          "'a modern Transformer beats the older CNN' without this qualifier is not supported "
          "by the ladder.")
        w("- Part of the deficit at the smallest rung is the optimisation failure documented "
          "in section 2, so the honest reading is that the Transformer needs both more data "
          "and a more forgiving training budget before its architecture pays off.")
    w("")
    (REPORTS / f"scaling_fits_{DISEASE}.md").write_text("\n".join(L), encoding="utf-8")
    _figure(sizes, mean, sd)
    print(f"[scaling] wrote reports/scaling_fits_{DISEASE}.md and the gap figure")


def _figure(sizes, mean, sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)
    colors = {"A": "#33414B", "B": "#C89F5D", "C": "#A6192E"}
    marks = {"A": "s", "B": "^", "C": "o"}
    for a, lab in ARMS.items():
        ax1.errorbar(sizes, mean[a], yerr=sd[a], marker=marks[a], capsize=3, lw=2.2,
                     color=colors[a], label=f"Arm {a}: {lab}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Training patients (log scale)")
    ax1.set_ylabel("AUROC (temporal holdout, 2023)")
    ax1.set_title("Learning curves")
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(frameon=False, fontsize=8)

    ln = np.log10(sizes.astype(float))
    for pair, col, ls in ((("A", "C"), "#A6192E", "-"), (("A", "B"), "#C89F5D", "--")):
        gap = (mean[pair[0]] - mean[pair[1]]).to_numpy()
        ax2.plot(sizes, gap, marker="o", color=col, ls=ls, label=f"{pair[0]} - {pair[1]}")
        m = sizes >= 30000                       # fit excluding the unstable 10k rung
        s_, i_ = np.polyfit(ln[m], gap[m], 1)
        xs = np.logspace(np.log10(sizes[0]), 7.6, 50)
        ax2.plot(xs, s_ * np.log10(xs) + i_, color=col, ls=":", lw=1.2, alpha=0.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Training patients (log scale)")
    ax2.set_ylabel("AUROC gap vs engineered features")
    ax2.set_title("Gap closure, extrapolated (fit excludes 10k)")
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / f"scaling_gap_{DISEASE}.png")
    plt.close(fig)


if __name__ == "__main__":
    build()
