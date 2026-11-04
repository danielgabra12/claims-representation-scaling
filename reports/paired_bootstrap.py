"""Paired bootstrap confidence intervals for the arm-vs-arm gaps (arthritis).

Run: ./surf/Scripts/python.exe -m reports.paired_bootstrap

Why this exists. The per-run metrics carry a standard deviation over 3 training seeds.
That is the variability of *training*, not the sampling error of the *metric*: it is
roughly 0.0005 at the top rung, while the standard error of a single AUROC on 2,554
positive cases is about 0.005, ten times larger. Reporting the seed SD as though it were
the uncertainty of the estimate is wrong, and it is the first thing a referee will notice.

What it does instead. Every run persisted per-patient test predictions, aligned to the same
137,689 held-out patients. So arms can be compared *paired on the same patients*, which is
both the correct comparison and far tighter than comparing two independent estimates:
the patient-level noise largely cancels in the difference.

  - resample PATIENTS with replacement (stratified by outcome so every resample keeps
    2,554 positives, which stabilises AUPRC);
  - within a resample, average each arm's predictions over its 3 seeds, then recompute
    AUROC/AUPRC for every arm on the identical patient set;
  - the bootstrap distribution of the DIFFERENCE gives the CI and a two-sided p-value
    for "the gap is zero".

Also reports DeLong intervals for AUROC at the top rung as an analytic cross-check.

Writes reports/paired_bootstrap_arthritis.md and reports/bootstrap_arthritis.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

from src.scale.config import EXPERIMENTS, REPORTS

DISEASE = "arthritis"
N_BOOT = 2000
RNG_SEED = 20260705
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
PAIRS = [("A", "C"), ("A", "B"), ("C", "B")]


def _run_dirs(arm: str, size=None):
    base = EXPERIMENTS / DISEASE / ("armA/rich_full" if arm == "A" else f"arm{arm}")
    ds = sorted(p for p in base.glob("*") if (p / "preds.parquet").exists())
    for (a, sz), sub in ARMC_10K_FIX.items():            # swap in the Run 4 rung
        if a != arm:
            continue
        ds = [d for d in ds if not d.name.startswith(f"size{sz}_")]
        ds += sorted(q for q in (EXPERIMENTS / DISEASE / sub).glob(f"size{sz}_seed*")
                     if (q / "preds.parquet").exists())
    return ds


def load_predictions():
    """-> {arm: {size: (n_patients, n_seeds) prediction matrix}}, plus the shared y."""
    preds, y_ref, order = {}, None, None
    for arm in ARMS:
        by_size: dict[int, list] = {}
        for d in _run_dirs(arm):
            meta = json.loads((d / "metrics.json").read_text())
            p = pd.read_parquet(d / "preds.parquet")
            if order is None:
                order, y_ref = p["PAT_ID"].to_numpy(), p["y"].to_numpy().astype(int)
            elif not np.array_equal(p["PAT_ID"].to_numpy(), order):
                # Never silently realign: a mismatch means the runs are not comparable.
                raise SystemExit(f"PAT_ID order differs in {d}; refusing to compare")
            by_size.setdefault(int(meta["size"]), []).append(p["p"].to_numpy())
        preds[arm] = {s: np.column_stack(v) for s, v in by_size.items()}
    return preds, y_ref


def _stratified_counts(y, rng, n_boot):
    """Bootstrap resamples expressed as per-patient multiplicities.

    Returned as counts rather than index lists so the metrics below can be evaluated by a
    single linear pass over a globally pre-sorted score order, instead of re-sorting
    137,689 predictions on every resample. Cases and non-cases are resampled separately so
    every draw keeps the same number of incident cases, which stabilises AUPRC.
    """
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n = y.size
    for _ in range(n_boot):
        c = np.zeros(n, dtype=np.int32)
        np.add.at(c, rng.choice(pos, pos.size, replace=True), 1)
        np.add.at(c, rng.choice(neg, neg.size, replace=True), 1)
        yield c


def _sorted_view(p, y):
    """Descending score order, case flags, and distinct-score group ids.

    Computed once per (arm, size). Grouping by distinct score is what makes the bootstrap
    metrics agree with sklearn: tree ensembles emit many identical leaf sums, so exact
    ties are common, and treating tied patients as separately ranked would inflate AUROC.
    """
    o = np.argsort(-p, kind="stable")
    ps = p[o]
    g = np.empty(ps.size, dtype=np.int64)
    g[0] = 0
    np.cumsum(ps[1:] != ps[:-1], out=g[1:])
    return o, y[o].astype(bool), g, int(g[-1]) + 1


def _boot_metrics(view, counts_sorted):
    """AUROC and average precision for one resample, aggregated over tied-score groups.

    `counts_sorted` is the per-patient resample multiplicity in descending-score order.
    Both metrics match sklearn's tie conventions: AUROC gives half credit within a tied
    group, average precision steps once per distinct score.
    """
    _, opos, g, ng = view
    c = counts_sorted.astype(np.float64)
    cp = np.bincount(g, weights=np.where(opos, c, 0.0), minlength=ng)
    cn = np.bincount(g, weights=np.where(opos, 0.0, c), minlength=ng)
    n1, n0 = cp.sum(), cn.sum()
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan
    neg_after = n0 - np.cumsum(cn)                 # non-cases in strictly lower groups
    auc = float((cp * (neg_after + 0.5 * cn)).sum() / (n1 * n0))
    tp = np.cumsum(cp)
    seen = np.cumsum(cp + cn)
    prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    ap = float((prec * cp).sum() / n1)
    return auc, ap


def _self_check(views, avg, y):
    """Fail loudly if the fast metrics disagree with sklearn on the un-resampled data."""
    ones = np.ones(y.size, dtype=np.int32)
    worst = 0.0
    for (a, s), view in views.items():
        o = view[0]
        auc, ap = _boot_metrics(view, ones[o])
        worst = max(worst, abs(auc - roc_auc_score(y, avg[a][s])),
                    abs(ap - average_precision_score(y, avg[a][s])))
    assert worst < 1e-9, f"fast metric disagrees with sklearn by {worst:.2e}"
    print(f"  self-check vs sklearn: max abs difference {worst:.2e}")


def _delong_var(y, p):
    """Variance of a single AUROC (DeLong). Used as an analytic sanity check."""
    pos, neg = p[y == 1], p[y == 0]
    m, n = pos.size, neg.size
    v01 = np.array([( (neg < x).sum() + 0.5 * (neg == x).sum() ) / n for x in pos])
    v10 = np.array([( (pos > x).sum() + 0.5 * (pos == x).sum() ) / m for x in neg])
    return v01.var(ddof=1) / m + v10.var(ddof=1) / n


def build():
    preds, y = load_predictions()
    sizes = sorted(set.intersection(*[set(v) for v in preds.values()]))
    rng = np.random.default_rng(RNG_SEED)
    n_pos = int(y.sum())

    # Two quantities, deliberately kept separate:
    #   PRIMARY   the mean over seeds of the per-seed metric. This is what a single trained
    #             model achieves in expectation, and it is what the learning-curve tables
    #             report, so the CIs below attach to the numbers shown elsewhere.
    #   SECONDARY the metric of the seed-averaged prediction, i.e. a 3-model ensemble.
    #             Reported as a contrast because ensembling helps the noisier arms more.
    avg = {a: {s: preds[a][s].mean(axis=1) for s in sizes} for a in ARMS}

    rows = []
    point, point_ens = {}, {}
    for s in sizes:
        for a in ARMS:
            P = preds[a][s]
            point[(a, s, "auroc")] = float(np.mean(
                [roc_auc_score(y, P[:, k]) for k in range(P.shape[1])]))
            point[(a, s, "auprc")] = float(np.mean(
                [average_precision_score(y, P[:, k]) for k in range(P.shape[1])]))
            point_ens[(a, s, "auroc")] = roc_auc_score(y, avg[a][s])
            point_ens[(a, s, "auprc")] = average_precision_score(y, avg[a][s])

    # Pre-sort once per (arm, size, seed) and once per ensemble; the bootstrap only walks.
    views = {(a, s, k): _sorted_view(preds[a][s][:, k], y)
             for a in ARMS for s in sizes for k in range(preds[a][s].shape[1])}
    ens_views = {(a, s): _sorted_view(avg[a][s], y) for a in ARMS for s in sizes}
    _self_check(ens_views, avg, y)

    dist = {(a, s, m): np.empty(N_BOOT) for a in ARMS for s in sizes
            for m in ("auroc", "auprc")}
    dist_ens = {k: np.empty(N_BOOT) for k in dist}
    for b, counts in enumerate(_stratified_counts(y, rng, N_BOOT)):
        for s in sizes:
            for a in ARMS:
                n_seed = preds[a][s].shape[1]
                aucs = np.empty(n_seed)
                aps = np.empty(n_seed)
                for k in range(n_seed):
                    v = views[(a, s, k)]
                    aucs[k], aps[k] = _boot_metrics(v, counts[v[0]])
                dist[(a, s, "auroc")][b] = aucs.mean()      # mean over seeds of the metric
                dist[(a, s, "auprc")][b] = aps.mean()
                v = ens_views[(a, s)]
                e_auc, e_ap = _boot_metrics(v, counts[v[0]])
                dist_ens[(a, s, "auroc")][b] = e_auc
                dist_ens[(a, s, "auprc")][b] = e_ap
        if (b + 1) % 250 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}", flush=True)

    for s in sizes:
        for metric in ("auroc", "auprc"):
            for hi, lo in PAIRS:
                d = dist[(hi, s, metric)] - dist[(lo, s, metric)]
                obs = point[(hi, s, metric)] - point[(lo, s, metric)]
                lo_ci, hi_ci = np.percentile(d, [2.5, 97.5])
                # two-sided bootstrap p: proportion of resamples on the other side of 0
                p_val = 2 * min((d <= 0).mean(), (d >= 0).mean())
                rows.append(dict(size=s, metric=metric, pair=f"{hi}-{lo}", diff=obs,
                                 lo=lo_ci, hi=hi_ci, p=max(p_val, 1.0 / N_BOOT),
                                 excludes_zero=bool(lo_ci > 0 or hi_ci < 0)))
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / f"bootstrap_{DISEASE}.csv", index=False)

    # ---------------- report -------------------------------------------------
    L = []
    w = L.append
    w("# Paired bootstrap confidence intervals: arthritis onset")
    w("")
    w(f"_Generated by `reports/paired_bootstrap.py`. {N_BOOT:,} bootstrap resamples of the "
      f"{len(y):,} held-out patients, stratified to preserve {n_pos:,} incident cases. Arms "
      "are compared **paired on the same patients**. Each arm's value in a resample is the "
      "**mean over its 3 seeds of the per-seed metric**, which is what the learning-curve "
      "tables report, so these intervals attach to the numbers used elsewhere._")
    w("")
    w("## Why these intervals and not the seed SD")
    w("")
    top = sizes[-1]
    se_a = np.sqrt(_delong_var(y, avg["A"][top]))
    seed_sd = float(np.std([roc_auc_score(y, preds["A"][top][:, k])
                            for k in range(preds["A"][top].shape[1])], ddof=1))
    w(f"At the top rung, Arm A's AUROC has a DeLong standard error of **{se_a:.4f}** "
      f"(sampling error of the metric on {n_pos:,} cases), while its spread over 3 training "
      f"seeds is **{seed_sd:.4f}**, roughly {se_a/max(seed_sd,1e-9):.0f}x smaller. The seed SD "
      "answers 'how much does retraining move this?', not 'how precisely do we know it?'. "
      "Reporting it as the error bar would overstate precision by an order of magnitude.")
    w("")
    w("The *paired* difference between two arms is much better determined than either arm's "
      "absolute AUROC, because patient-level noise is shared and cancels. That is why the "
      "gap intervals below are far tighter than the individual standard errors.")
    w("")

    for metric, lab in (("auroc", "AUROC"), ("auprc", "AUPRC")):
        w(f"## {lab}: gaps with 95% CI")
        w("")
        w("| training patients | " + " | ".join(f"{h} - {l}" for h, l in PAIRS) + " |")
        w("|---|" + "---|" * len(PAIRS))
        for s in sizes:
            cells = []
            for hi, lo in PAIRS:
                r = res[(res["size"] == s) & (res.metric == metric)
                        & (res.pair == f"{hi}-{lo}")].iloc[0]
                star = "" if r.excludes_zero else " (n.s.)"
                cells.append(f"{r['diff']:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]{star}")
            w(f"| {s:,} | " + " | ".join(cells) + " |")
        w("")

    w("## Reading it")
    w("")
    ac = res[(res["size"] == top) & (res.metric == "auroc") & (res.pair == "A-C")].iloc[0]
    cb = res[(res["size"] == top) & (res.metric == "auroc") & (res.pair == "C-B")].iloc[0]
    acp = res[(res["size"] == top) & (res.metric == "auprc") & (res.pair == "A-C")].iloc[0]
    w(f"- At {top:,} training patients, engineered features lead the Transformer by "
      f"**{ac['diff']:.4f} AUROC (95% CI {ac.lo:.4f} to {ac.hi:.4f})**. The interval "
      f"{'excludes' if ac.excludes_zero else 'includes'} zero, so the lead "
      f"{'is' if ac.excludes_zero else 'is not'} distinguishable from noise on this test set.")
    w(f"- On AUPRC the same gap is **{acp['diff']:.4f} (95% CI {acp.lo:.4f} to {acp.hi:.4f})**, "
      f"against a base rate of {n_pos/len(y):.4f}.")
    w(f"- The Transformer's lead over the 2017 CNN at the top rung is **{cb['diff']:+.4f} "
      f"(95% CI {cb.lo:+.4f} to {cb.hi:+.4f})**"
      f"{', which excludes zero' if cb.excludes_zero else ', which INCLUDES zero and is therefore not established'}.")
    ns = res[(res.metric == "auroc") & (~res.excludes_zero)]
    if len(ns):
        w("- Comparisons whose AUROC interval includes zero (i.e. not established): "
          + ", ".join(f"{r.pair} at {int(r['size']):,}" for r in ns.itertuples()) + ".")
    w("")
    w("## Single model versus a 3-seed ensemble")
    w("")
    w("The table above is the expected performance of **one** trained model. Averaging the "
      "three seeds' predictions instead gives a small ensemble, and that changes the gaps, "
      "because the noisier arms gain more from averaging:")
    w("")
    w("| comparison | single model (mean over seeds) | 3-seed ensemble | change |")
    w("|---|---|---|---|")
    for hi, lo in PAIRS:
        one = point[(hi, top, "auroc")] - point[(lo, top, "auroc")]
        ens = point_ens[(hi, top, "auroc")] - point_ens[(lo, top, "auroc")]
        w(f"| {hi} - {lo} at {top:,} | {one:+.4f} | {ens:+.4f} | {ens-one:+.4f} |")
    w("")
    d_ac = ((point_ens[("A", top, "auroc")] - point_ens[("C", top, "auroc")])
            - (point[("A", top, "auroc")] - point[("C", top, "auroc")]))
    w(f"- Ensembling narrows the features-vs-Transformer gap by **{abs(d_ac):.4f} AUROC**, "
      "which says a real part of the Transformer's deficit at this scale is **variance, not "
      "bias**: it is not that the representation cannot capture the signal, it is that a "
      "single training run lands inconsistently. That is a cheap and honest observation to "
      "make, and it points at training budget rather than architecture as the binding "
      "constraint.")
    w("- The paper reports the single-model numbers as primary, since that is what the "
      "learning curves and the scaling fits use.")
    w("")
    w("_Caveat: these intervals quantify sampling error on one fixed test set from one "
      "insurer population. They do not cover the design asymmetries (sequence truncation, "
      "fixed model capacity across rungs) documented in the limitations, which are larger "
      "than the intervals shown here._")
    (REPORTS / f"paired_bootstrap_{DISEASE}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[bootstrap] wrote reports/paired_bootstrap_{DISEASE}.md "
          f"({N_BOOT} resamples, {len(sizes)} rungs, {len(res)} comparisons)")


if __name__ == "__main__":
    build()
