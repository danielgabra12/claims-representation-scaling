"""Publication figures for the ML4H submission.

Run: ./surf/Scripts/python.exe -m reports.figures_paper

Figure 1  Learning curves for the three arms across the ladder. Error bars are the seed range
          and are labelled as such: they are training variability, NOT the sampling error of the
          metric, and the paper is explicit about that distinction (section 4.2).
Figure 2  Gap closure with the extrapolation. Error bars are the PAIRED bootstrap intervals,
          which is the correct uncertainty for a difference measured on the same patients. The
          measured range and the extrapolated range are drawn differently, because the
          extrapolation is an assumption and should not read as data.

Both use the corrected 10k Arm C rung (section 4.1), a colourblind-safe palette, and are written
at 400 DPI in both PNG and PDF.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scale.config import EXPERIMENTS, REPORTS
from reports.scaling_analysis import ARMC_10K_FIX, _gap_fit

DISEASE = "arthritis"
OUT = REPORTS / "figures"
# Okabe-Ito: colourblind-safe, and distinguishable in greyscale by lightness order.
COL = {"A": "#000000", "B": "#E69F00", "C": "#0072B2"}
MARK = {"A": "s", "B": "^", "C": "o"}
NAME = {"A": "Arm A: engineered features (GBDT)",
        "B": "Arm B: code sequence (CNN)",
        "C": "Arm C: code sequence (Transformer)"}


def _arm(arm):
    sub = "armA/rich_full" if arm == "A" else f"arm{arm}"
    rows = [json.loads(p.read_text())
            for p in (EXPERIMENTS / DISEASE / sub).glob("*/metrics.json")]
    for (a, size), fix in ARMC_10K_FIX.items():          # corrected 10k rung
        if a == arm:
            rows = [r for r in rows if r["size"] != size]
            rows += [json.loads(q.read_text()) for q in
                     (EXPERIMENTS / DISEASE / fix).glob(f"size{size}_seed*/metrics.json")]
    d = pd.DataFrame(rows)
    return d.groupby("size").auroc.agg(["mean", "min", "max", "count"])


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8,
                         "xtick.direction": "out", "ytick.direction": "out",
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})

    arms = {a: _arm(a) for a in "ABC"}
    sizes = arms["A"].index.to_numpy()
    boot = pd.read_csv(REPORTS / f"bootstrap_{DISEASE}.csv")
    boot = boot[boot.metric == "auroc"].set_index(["pair", "size"])

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- fig 1
    fig, ax = plt.subplots(figsize=(6.4, 2.5), dpi=400)
    for a in "ABC":
        g = arms[a]
        lo = g["mean"] - g["min"]
        hi = g["max"] - g["mean"]
        ax.errorbar(sizes, g["mean"], yerr=[lo, hi], marker=MARK[a], ms=4.5, lw=1.6,
                    capsize=2.5, elinewidth=0.9, color=COL[a], label=NAME[a], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Training patients (log scale)")
    ax.set_ylabel("AUROC, held-out 2023 test set")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_learning_curves.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig 2
    fig, ax = plt.subplots(figsize=(6.4, 2.5), dpi=400)
    xmax = 3e7
    fit_mask = sizes >= 30000                      # 10k excluded: least stable rung
    for pair, col, ls, lab in (("A-C", COL["C"], "-", "Features minus Transformer"),
                               ("A-B", COL["B"], "-", "Features minus CNN")):
        d = np.array([boot.loc[(pair, s), "diff"] for s in sizes])
        lo = np.array([boot.loc[(pair, s), "lo"] for s in sizes])
        hi = np.array([boot.loc[(pair, s), "hi"] for s in sizes])
        ax.errorbar(sizes, d, yerr=[d - lo, hi - d], marker="o", ms=4, lw=1.6, ls=ls,
                    capsize=2.5, elinewidth=0.9, color=col, label=lab, zorder=3)
        slope, parity = _gap_fit(sizes[fit_mask], d[fit_mask])
        b = d[fit_mask][0] - slope * np.log10(sizes[fit_mask][0])
        xf = np.logspace(np.log10(sizes[fit_mask][0]), np.log10(sizes[-1]), 30)
        ax.plot(xf, slope * np.log10(xf) + b, color=col, lw=1.0, ls="-", alpha=0.45, zorder=2)
        xe = np.logspace(np.log10(sizes[-1]), np.log10(xmax), 60)
        ax.plot(xe, slope * np.log10(xe) + b, color=col, lw=1.2, ls=(0, (4, 2)), zorder=2)
        if np.isfinite(parity) and parity < xmax:
            ax.plot([parity], [0], marker="v", ms=6, color=col, zorder=4, clip_on=False)

    # the extrapolated region is shaded so it never reads as data
    ax.axvspan(sizes[-1], xmax, color="#BBBBBB", alpha=0.22, lw=0, zorder=0)
    ax.axhline(0, color="#000000", lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(sizes[0] * 0.75, xmax)
    ax.set_xlabel("Training patients (log scale)")
    ax.set_ylabel("AUROC gap vs engineered features")
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.5, zorder=0)
    ax.legend(loc="upper right", fontsize=7.5)
    # Region labels run along the bottom so they cannot collide with the legend; the
    # explanatory note sits below the axes rather than inside the data area.
    ax.annotate("measured", xy=(np.sqrt(sizes[0] * sizes[-1]), 0.06),
                xycoords=("data", "axes fraction"), ha="center", va="bottom",
                fontsize=7.5, color="#333333")
    ax.annotate("extrapolated", xy=(np.sqrt(sizes[-1] * xmax), 0.06),
                xycoords=("data", "axes fraction"), ha="center", va="bottom",
                fontsize=7.5, color="#333333")
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_gap_closure.{ext}", bbox_inches="tight")
    plt.close(fig)

    for f in ("fig1_learning_curves", "fig2_gap_closure"):
        p = OUT / f"{f}.png"
        print(f"  {f}: {p.stat().st_size/1024:.0f} KB png + pdf")
    print(f"[figures] written to {OUT}")


if __name__ == "__main__":
    build()
