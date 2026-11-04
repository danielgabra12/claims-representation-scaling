"""Stage 6 — Arm A: engineered features -> gradient-boosted trees (XGBoost).

Run: python -m src.scale.models.arm_a            # arthritis

Trains Arm A at every (size, seed) on the nested ladder, evaluates on the FIXED
temporal test set, twice per cell: full features and utilization-removed (the
access-confound ablation). Writes one experiments/ dir per run (config.json +
metrics.json) and an aggregate reports/stage6_<disease>_armA.md + learning-curve fig.

Leakage safety: features are already pre-index (Stage 3). The early-stopping
validation slice is carved from the TRAINING subset only; the test set is untouched
during fitting. No test-derived statistic enters training.

Compute budget (fixed, documented for the A-vs-B fairness contract):
  XGBoost hist, max 800 rounds, early stop 40 on a 10% train-carved val AUC,
  depth 6, lr 0.05, subsample/colsample 0.8, min_child_weight 5, no pos reweight.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb

from src.scale.config import DISEASES, DATA_PROCESSED, EXPERIMENTS, REPORTS, seed_for
from src.scale.scaling.ladder import ladder_sizes, train_subset_ids, test_ids

def _xgb_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


XGB_PARAMS = dict(max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5,
                  tree_method="hist", device=_xgb_device(),  # GPU on Colab, CPU locally
                  eval_metric="auc", n_jobs=-1)
NUM_ROUNDS = 800
EARLY_STOP = 40
VAL_FRAC = 0.10


def _load(disease: str):
    fpq = (DATA_PROCESSED / f"{disease}_features.parquet").as_posix()
    meta = json.loads((DATA_PROCESSED / f"{disease}_features_meta.json").read_text())
    con = __import__("duckdb").connect()
    df = con.execute(f"SELECT * FROM read_parquet('{fpq}')").df()
    df = df.set_index("PAT_ID")
    return df, meta


def _fit_eval(df, feat_cols, train_ids, test_ids_, seed):
    rng = np.random.default_rng(seed)
    tr = np.array(train_ids, dtype=object)
    rng.shuffle(tr)
    n_val = int(len(tr) * VAL_FRAC)
    val_ids, fit_ids = tr[:n_val], tr[n_val:]

    Xf, yf = df.loc[fit_ids, feat_cols], df.loc[fit_ids, "label"]
    Xv, yv = df.loc[val_ids, feat_cols], df.loc[val_ids, "label"]
    Xt, yt = df.loc[test_ids_, feat_cols], df.loc[test_ids_, "label"]

    clf = xgb.XGBClassifier(n_estimators=NUM_ROUNDS, early_stopping_rounds=EARLY_STOP,
                            random_state=seed, **XGB_PARAMS)
    clf.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    p = clf.predict_proba(Xt)[:, 1]
    # Persist per-patient test predictions so paired bootstrap / DeLong intervals on the
    # A-vs-B-vs-C gaps can be computed later without re-training. Row order == test_ids_.
    preds = pd.DataFrame({"PAT_ID": list(test_ids_),
                          "y": yt.to_numpy().astype("int8"),
                          "p": p.astype("float32")})
    return {
        "auroc": float(roc_auc_score(yt, p)),
        "auprc": float(average_precision_score(yt, p)),
        "brier": float(brier_score_loss(yt, p)),
        "best_iteration": int(clf.best_iteration),
        "n_fit": int(len(fit_ids)), "n_val": int(len(val_ids)),
        "pos_rate_train": float(yf.mean()),
    }, preds


def build(disease: str = "arthritis", tag: str = "", sizes=None, n_seeds: int = 3,
          featuresets_only=None):
    """Train Arm A across the ladder.

    `tag` routes output to experiments/<disease>/armA<tag>/ so an ablation never overwrites
    the headline runs; `sizes` restricts the rungs; `featuresets_only` restricts which
    feature sets are run (the ablations only need `rich_full`).
    """
    df, meta = _load(disease)
    # Three feature sets span both ablation axes (see reports/fairness_contract.md):
    #   rich_full   : the fair baseline (all families)
    #   rich_noutil : rich minus utilization+temporal (access-confound ablation)
    #   thin_full   : the original 27 features (richness regression baseline ~0.828)
    featuresets = (
        ("thin_full", meta["thin"]),
        ("rich_noutil", meta["clinical"]),
        ("rich_full", meta["features"]),
    )
    if featuresets_only:
        featuresets = tuple(f for f in featuresets if f[0] in featuresets_only)
    sizes = ladder_sizes(disease) if sizes is None else list(sizes)
    test = test_ids(disease)

    records = []
    for featset_name, feat_cols in featuresets:
        for size in sizes:
            for k in range(n_seeds):
                run = (EXPERIMENTS / disease / f"armA{tag}" / featset_name
                       / f"size{size}_seed{k}")
                # resume: a run counts as done only if BOTH metrics and predictions landed
                if (run / "metrics.json").exists() and (run / "preds.parquet").exists():
                    records.append(json.loads((run / "metrics.json").read_text()))
                    continue
                sub = train_subset_ids(disease, size, k)
                m, preds = _fit_eval(df, feat_cols, sub, test, seed_for(k))
                m.update(disease=disease, arm="A", featureset=featset_name,
                         size=size, seed=k, n_features=len(feat_cols))
                run.mkdir(parents=True, exist_ok=True)
                preds.to_parquet(run / "preds.parquet", index=False)
                (run / "config.json").write_text(json.dumps(
                    {"disease": disease, "arm": "A", "featureset": featset_name,
                     "tag": tag, "size": size, "seed": k, "xgb_params": XGB_PARAMS,
                     "num_rounds": NUM_ROUNDS, "early_stop": EARLY_STOP,
                     "features": feat_cols}, indent=2), encoding="utf-8")
                (run / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
                records.append(m)
                print(f"  {featset_name:8s} size={size:>7,} seed={k} "
                      f"AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f}")

    res = pd.DataFrame(records)
    res.to_csv(REPORTS / f"stage6_{disease}_armA{tag}_results.csv", index=False)
    if not tag:
        # The learning-curve report/figure describe the headline configuration only.
        _report(disease, res, sizes)
        _figure(disease, res, sizes)
    print(f"[arm_a] {disease}: {len(records)} runs -> reports/stage6_{disease}_armA.md")


def _agg(res):
    g = res.groupby(["featureset", "size"]).agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"), brier_mean=("brier", "mean")).reset_index()
    return g


def _cell(g, fs, s):
    return g[(g.featureset == fs) & (g["size"] == s)].iloc[0]


def _report(disease, res, sizes):
    g = _agg(res)
    L = []; w = L.append
    w(f"# Stage 6 — Arm A (GBDT) across the scaling ladder: {DISEASES[disease]['label']}")
    w("")
    w("_Engineered features -> XGBoost. Fixed temporal test set. Mean +/- SD over 3 seeds. "
      "Three feature sets (see reports/fairness_contract.md): `thin` (original 27), "
      "`rich_full` (fair baseline, all families), `rich_noutil` (rich minus "
      "utilization+temporal = access-confound ablation)._")
    w("")
    w("## Learning curve (temporal holdout, AUROC mean ± SD)")
    w("| size | thin_full | rich_full | rich_noutil | Δ richness | Δ util |")
    w("|---|---|---|---|---|---|")
    for s in sizes:
        t = _cell(g, "thin_full", s); f = _cell(g, "rich_full", s); nu = _cell(g, "rich_noutil", s)
        w(f"| {s:,} | {t.auroc_mean:.4f} ± {t.auroc_sd:.4f} | "
          f"{f.auroc_mean:.4f} ± {f.auroc_sd:.4f} | {nu.auroc_mean:.4f} ± {nu.auroc_sd:.4f} | "
          f"{f.auroc_mean - t.auroc_mean:+.4f} | {f.auroc_mean - nu.auroc_mean:+.4f} |")
    w("")
    w("- **Δ richness** = rich_full − thin_full: how much the bag-of-codes / drug-class "
      "enrichment bought (the fairness upgrade).")
    w("- **Δ util** = rich_full − rich_noutil: the portion attributable to the "
      "healthcare-access confound rather than clinical content.")
    w("")
    w("## AUPRC / calibration (Brier), rich_full")
    w("| size | AUPRC | Brier |")
    w("|---|---|---|")
    for s in sizes:
        f = _cell(g, "rich_full", s)
        w(f"| {s:,} | {f.auprc_mean:.4f} | {f.brier_mean:.4f} |")
    w("")
    smax = sizes[-1]
    tmax = _cell(g, "thin_full", smax); fmax = _cell(g, "rich_full", smax)
    numax = _cell(g, "rich_noutil", smax)
    guard = "PASS" if fmax.auroc_mean >= tmax.auroc_mean else "REGRESSION"
    w("## Reading it")
    w(f"- At the largest local size ({smax:,}): thin **{tmax.auroc_mean:.3f}** -> "
      f"rich_full **{fmax.auroc_mean:.3f}** (Δ richness {fmax.auroc_mean - tmax.auroc_mean:+.3f}). "
      f"Regression guard vs thin baseline: **{guard}**.")
    w(f"- Removing utilization+temporal from rich drops it to **{numax.auroc_mean:.3f}** "
      f"(Δ util {fmax.auroc_mean - numax.auroc_mean:+.3f}) — the access-confound share.")
    w("- `rich_full` is the Arm A half of the crossover plot; the Arm B curve (Stage 7) "
      "overlays on the same axes. Local sizes stop at the train-pool max; >=1M needs the "
      "full extraction.")
    w("")
    w(f"Per-run configs/metrics: `experiments/{disease}/armA/`. "
      f"Raw table: `reports/stage6_{disease}_armA_results.csv`.")
    (REPORTS / f"stage6_{disease}_armA.md").write_text("\n".join(L), encoding="utf-8")


def _figure(disease, res, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = _agg(res)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for name, color, lab in (("rich_full", "#1b9e77", "Rich (fair baseline)"),
                             ("rich_noutil", "#d95f02", "Rich − utilization (ablation)"),
                             ("thin_full", "#7570b3", "Thin (original 27)")):
        d = g[g.featureset == name].sort_values("size")
        ax.errorbar(d["size"], d["auroc_mean"], yerr=d["auroc_sd"], marker="o",
                    capsize=3, color=color, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Training set size (patients, log scale)")
    ax.set_ylabel("AUROC (temporal holdout)")
    ax.set_title(f"Arm A learning curve — {DISEASES[disease]['label']}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    (REPORTS / "figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / f"stage6_{disease}_armA.png")
    plt.close(fig)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")
