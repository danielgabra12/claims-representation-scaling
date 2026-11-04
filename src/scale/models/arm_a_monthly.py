"""Stage 9 - Arm A-M: monthly-resolution bag -> XGBoost (the information-equivalence control).

Run: ./surf/Scripts/python.exe -m src.scale.models.arm_a_monthly arthritis

Identical to Arm A in every respect except that count features are split into 12 monthly bins
(see src/scale/features/monthly.py). Same cohort, same frozen vocabulary, same nested ladder,
same seeds, same train-carved validation split, same fixed test set, and the SAME XGBoost
configuration with no retuning.

Two variants are run at every (size, seed):

  `monthly`   the 9,672-column monthly-binned matrix. This is Arm A-M.
  `agg_ctrl`  the same matrix with the monthly bins summed back to per-window totals, which
              reproduces Arm A's 828 features exactly. This exists because the monthly matrix
              is sparse and XGBoost treats structural zeros in a CSR as MISSING rather than as
              zero, which is a semantic difference from Arm A's dense matrix. Running the
              aggregated control through the identical sparse code path isolates binning from
              that difference: `monthly` minus `agg_ctrl` is a clean binning contrast, and
              `agg_ctrl` versus the committed dense Arm A results measures how much the
              sparse/missing semantics matter on its own.

Writes experiments/<disease>/armA_monthly/<variant>/size{n}_seed{k}/ with config.json,
metrics.json and preds.parquet, plus reports/stage9_<disease>_armAM_results.csv.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb

from src.scale.config import DATA_PROCESSED, DISEASES, EXPERIMENTS, REPORTS, seed_for
from src.scale.models.arm_a import EARLY_STOP, NUM_ROUNDS, VAL_FRAC, XGB_PARAMS
from src.scale.scaling.ladder import ladder_sizes, test_ids, train_subset_ids


def _load(disease: str):
    X = sparse.load_npz(DATA_PROCESSED / f"{disease}_monthly.npz").tocsr()
    rows = np.load(DATA_PROCESSED / f"{disease}_monthly_rows.npy", allow_pickle=True)
    meta = json.loads((DATA_PROCESSED / f"{disease}_monthly_meta.json").read_text())
    lab = pd.read_parquet(DATA_PROCESSED / f"{disease}_features.parquet",
                          columns=["PAT_ID", "label"]).set_index("PAT_ID")
    y = lab.loc[rows, "label"].to_numpy().astype(np.int8)
    pos = {p: i for i, p in enumerate(rows)}
    return X, y, pos, meta


# COUNT(DISTINCT code) features are NOT additive across bins: a code appearing in two months
# contributes 2 to the summed monthly version but 1 to Arm A's window-level count. Summing the
# bins therefore cannot reproduce these three columns, so the control takes their true
# window-level values directly from the Arm A matrix. Everything else pools exactly (verified
# to 0 absolute difference on 825 of 828 columns).
NON_ADDITIVE = ["n_dx_codes", "n_ndc", "n_proc_codes"]


def _pool(X, meta, nbins, disease=None, rows=None):
    """Pool the 12 monthly bins down to `nbins` equal bins (12, 4, 2 or 1).

    A single sparse right-multiply by a 0/1 pooling matrix, so every coarser binning is a
    re-aggregation of the one matrix rather than a rebuild. nbins=1 reproduces Arm A's 828
    columns exactly and is the control.

    Column layout of the result: the 24 unbinned columns, then n_code_slots * nbins, then
    n_util * nbins.
    """
    assert 12 % nbins == 0, "nbins must divide 12"
    nm = meta["n_months"]
    per = nm // nbins
    off_c, off_u = meta["off_codes"], meta["off_util"]
    n_slots, n_util = meta["n_code_slots"], len(meta["utilization"])
    n_unb = len(meta["unbinned"])
    n_out = n_unb + (n_slots + n_util) * nbins

    cols = np.empty(meta["n_cols"], dtype=np.int64)
    cols[:n_unb] = np.arange(n_unb)                                # unbinned pass through
    m_of = np.arange(nm) // per                                    # month -> bin
    cols[off_c:off_c + n_slots * nm] = (
        n_unb + np.repeat(np.arange(n_slots), nm) * nbins + np.tile(m_of, n_slots))
    cols[off_u:off_u + n_util * nm] = (
        n_unb + n_slots * nbins + np.repeat(np.arange(n_util), nm) * nbins
        + np.tile(m_of, n_util))
    P = sparse.csr_matrix(
        (np.ones(meta["n_cols"], dtype=np.float32), cols, np.arange(meta["n_cols"] + 1)),
        shape=(meta["n_cols"], n_out))
    A = (X @ P).tocsr()

    if nbins == 1 and disease is not None and rows is not None:
        # COUNT(DISTINCT code) features are NOT additive across bins: a code appearing in two
        # months contributes 2 to the summed monthly version but 1 to Arm A's window-level
        # count. Summing therefore cannot reproduce these three columns, so the control takes
        # their true window-level values. Everything else pools exactly (verified to 0
        # absolute difference on 825 of 828 columns).
        A = A.tolil()
        true = pd.read_parquet(DATA_PROCESSED / f"{disease}_features.parquet",
                               columns=["PAT_ID"] + NON_ADDITIVE).set_index("PAT_ID")
        true = true.loc[rows]
        util0 = n_unb + n_slots
        for name in NON_ADDITIVE:
            j = util0 + meta["utilization"].index(name)
            A[:, j] = true[name].to_numpy(dtype=np.float32).reshape(-1, 1)
        A = A.tocsr()
    return A


def _aggregate(X, meta, disease=None, rows=None):
    """Back-compatible alias: pool all the way to Arm A's 828 columns."""
    return _pool(X, meta, 1, disease, rows)


def _fit_eval(X, y, pos, train_ids, test_ids_, seed):
    rng = np.random.default_rng(seed)
    tr = np.array(train_ids, dtype=object)
    rng.shuffle(tr)                                    # same val carve rule as every other arm
    n_val = int(len(tr) * VAL_FRAC)
    val_ids, fit_ids = tr[:n_val], tr[n_val:]

    ifit = np.fromiter((pos[p] for p in fit_ids), dtype=np.int64, count=len(fit_ids))
    ival = np.fromiter((pos[p] for p in val_ids), dtype=np.int64, count=len(val_ids))
    itest = np.fromiter((pos[p] for p in test_ids_), dtype=np.int64, count=len(test_ids_))

    clf = xgb.XGBClassifier(n_estimators=NUM_ROUNDS, early_stopping_rounds=EARLY_STOP,
                            random_state=seed, **XGB_PARAMS)
    clf.fit(X[ifit], y[ifit], eval_set=[(X[ival], y[ival])], verbose=False)
    p = clf.predict_proba(X[itest])[:, 1]
    yt = y[itest]
    preds = pd.DataFrame({"PAT_ID": list(test_ids_), "y": yt.astype("int8"),
                          "p": p.astype("float32")})
    return {
        "auroc": float(roc_auc_score(yt, p)),
        "auprc": float(average_precision_score(yt, p)),
        "brier": float(brier_score_loss(yt, p)),
        "best_iteration": int(clf.best_iteration),
        "n_fit": int(len(fit_ids)), "n_val": int(len(val_ids)),
        "pos_rate_train": float(y[ifit].mean()),
    }, preds


# variant name -> number of temporal bins. 12 is Arm A-M; 1 is the exact Arm A control.
VARIANT_BINS = {"monthly": 12, "bin4": 4, "bin2": 2, "agg_ctrl": 1}


def build(disease: str = "arthritis", variants=("monthly", "agg_ctrl"), sizes=None,
          n_seeds: int = 3):
    X, y, pos, meta = _load(disease)
    rows = np.load(DATA_PROCESSED / f"{disease}_monthly_rows.npy", allow_pickle=True)
    mats = {}
    for v in variants:
        nb = VARIANT_BINS[v]
        mats[v] = X if nb == 12 else _pool(X, meta, nb, disease, rows)
    for k, m in mats.items():
        print(f"[armA-M] {k}: {m.shape} nnz={m.nnz:,}", flush=True)
    sizes = ladder_sizes(disease) if sizes is None else list(sizes)
    test = test_ids(disease)

    records = []
    for variant in variants:
        M = mats[variant]
        for size in sizes:
            for k in range(n_seeds):
                run = (EXPERIMENTS / disease / "armA_monthly" / variant
                       / f"size{size}_seed{k}")
                if (run / "metrics.json").exists() and (run / "preds.parquet").exists():
                    records.append(json.loads((run / "metrics.json").read_text()))
                    continue
                sub = train_subset_ids(disease, size, k)
                t0 = time.time()
                m, preds = _fit_eval(M, y, pos, sub, test, seed_for(k))
                m.update(disease=disease, arm="A-M", variant=variant, size=size, seed=k,
                         n_features=int(M.shape[1]), seconds=round(time.time() - t0, 1))
                run.mkdir(parents=True, exist_ok=True)
                preds.to_parquet(run / "preds.parquet", index=False)
                (run / "config.json").write_text(json.dumps(
                    {"disease": disease, "arm": "A-M", "variant": variant, "size": size,
                     "seed": k, "xgb_params": XGB_PARAMS, "num_rounds": NUM_ROUNDS,
                     "early_stop": EARLY_STOP, "n_features": int(M.shape[1]),
                     "n_months": meta["n_months"]}, indent=2), encoding="utf-8")
                (run / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
                records.append(m)
                print(f"  {variant:9s} size={size:>9,} seed={k} AUROC={m['auroc']:.4f} "
                      f"AUPRC={m['auprc']:.4f} (it{m['best_iteration']}, {m['seconds']}s)",
                      flush=True)

    res = pd.DataFrame(records)
    res.to_csv(REPORTS / f"stage9_{disease}_armAM_results.csv", index=False)
    print(f"[armA-M] {disease}: {len(records)} runs "
          f"-> reports/stage9_{disease}_armAM_results.csv", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")
