"""Stage 7 — Arm B: ordered code sequences -> 1D CNN (Deepr-style; the crossover payload).

Run: python -m src.scale.models.arm_b            # arthritis

Trains Arm B at every (size, seed) on the SAME nested ladder, the SAME train-carved
validation split, and the SAME fixed temporal test set as Arm A, so the A-vs-B curves
overlay as a fair crossover (reports/fairness_contract.md). Arm B sees exactly Arm A's
information — the same frozen code vocabulary plus the same 7 demographics — but in real
month order instead of collapsed to counts. Any gap is representation, not data.

Model: Embedding + learned positional encoding -> multi-kernel 1D CNN (Deepr-style;
Nguyen et al. 2017) -> masked global max-pool -> concat demographics -> MLP -> logit.
Architecture history on this CPU-only box: a BiGRU was ~15 h/run (RNNs are sequential
over timesteps) and a Transformer ~13 h/run (its padding-mask path falls off PyTorch's
CPU fast path). Conv1d is pure batched gemm and runs in minutes. It still reads ORDER —
local n-gram patterns of codes plus positional embeddings for recency — so it remains a
genuine sequence-representation contrast to Arm A's order-free bag. Compute budget is
fixed and documented as the Arm B side of the contract.

Writes one experiments/<disease>/armB/size{size}_seed{k}/ dir per run (config+metrics),
reports/stage7_<disease>_armB.md, the results CSV, and the crossover overlay figure.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.scale.config import DISEASES, DATA_PROCESSED, EXPERIMENTS, REPORTS, seed_for
from src.scale.scaling.ladder import ladder_sizes, train_subset_ids, test_ids

DEMO_COLS = ["age_norm", "is_female", "is_male", "reg_mw", "reg_ne", "reg_s", "reg_w"]
MAX_LEN = 200                       # matches the sequence builder's truncation
_SEQ_SUFFIX = ""                    # "" = the len-200 sequences the headline runs used

# --- fixed compute budget (Arm B side of the fairness contract) --------------
D_MODEL, KERNELS, N_FILTERS, MLP_HID, DROPOUT = 64, (3, 5), 64, 64, 0.2
BATCH, LR, MAX_EPOCHS, PATIENCE, VAL_FRAC = 512, 1e-3, 25, 4, 0.10
COSINE = False             # headline runs: constant LR. Run 2 turns this on.
HYPER = dict(d_model=D_MODEL, kernels=list(KERNELS), n_filters=N_FILTERS,
             mlp_hidden=MLP_HID, dropout=DROPOUT, max_len=MAX_LEN, batch=BATCH,
             lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, val_frac=VAL_FRAC,
             optimizer="adam", cosine=COSINE, loss="bce", pos_reweight=False)

torch.set_num_threads(8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU on Colab


class SeqDataset(Dataset):
    def __init__(self, ids, seqs, demo, label):
        self.seq = [seqs[i] for i in ids]
        self.demo = [demo[i] for i in ids]
        self.y = [label[i] for i in ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, k):
        return self.seq[k], self.demo[k], self.y[k]


def _collate(batch):
    seqs, demos, ys = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence([torch.from_numpy(s).long() for s in seqs],
                          batch_first=True, padding_value=0)
    demo = torch.tensor(np.stack(demos), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return padded, lens, demo, y


class SeqCNN(nn.Module):
    """Deepr-style multi-kernel 1D CNN over the code sequence (+ positional embedding)."""
    def __init__(self, vocab_size, n_demo):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.convs = nn.ModuleList(
            nn.Conv1d(D_MODEL, N_FILTERS, k, padding=k // 2) for k in KERNELS)
        self.head = nn.Sequential(
            nn.Linear(N_FILTERS * len(KERNELS) + n_demo, MLP_HID), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(MLP_HID, 1))

    def forward(self, x, lens, demo):
        L = x.size(1)
        pos = torch.arange(L, device=x.device).clamp(max=MAX_LEN - 1)[None, :]
        h = (self.emb(x) + self.pos(pos)).transpose(1, 2)      # (B, D, L)
        padmask = (torch.arange(L, device=x.device)[None, :] >= lens[:, None])  # (B,L)
        feats = []
        for conv in self.convs:
            c = torch.relu(conv(h))[..., :L]                   # (B, F, L)
            c = c.masked_fill(padmask.unsqueeze(1), float("-inf"))
            feats.append(c.max(dim=2).values)                  # global max-pool
        pooled = torch.cat(feats, dim=1)                       # (B, F*len(KERNELS))
        return self.head(torch.cat([pooled, demo], dim=1)).squeeze(1)


def _load(disease: str):
    df = pd.read_parquet(
        DATA_PROCESSED / f"{disease}_sequences{_SEQ_SUFFIX}.parquet").set_index("PAT_ID")
    vocab = json.loads(
        (DATA_PROCESSED / f"{disease}_sequences{_SEQ_SUFFIX}_vocab.json").read_text())
    seqs = {i: np.asarray(s, dtype=np.int64) for i, s in df["seq"].items()}
    demo_mat = df[["age_at_index", "is_female", "is_male",
                   "reg_mw", "reg_ne", "reg_s", "reg_w"]].to_numpy(dtype=np.float32).copy()
    demo_mat[:, 0] = demo_mat[:, 0] / 100.0                    # age -> ~[0.18, 0.89]
    demo = {i: demo_mat[k] for k, i in enumerate(df.index)}
    label = df["label"].astype(float).to_dict()
    return seqs, demo, label, int(vocab["vocab_size"])


def _loader(ids, seqs, demo, label, shuffle):
    return DataLoader(SeqDataset(ids, seqs, demo, label), batch_size=BATCH,
                      shuffle=shuffle, collate_fn=_collate, num_workers=0)


@torch.no_grad()
def _predict(model, loader):
    model.eval()
    ps, ys = [], []
    for x, lens, demo, y in loader:
        x, lens, demo = x.to(DEVICE), lens.to(DEVICE), demo.to(DEVICE)
        ps.append(torch.sigmoid(model(x, lens, demo)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def _fit_eval(seqs, demo, label, vocab_size, train_ids, test_ids_, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)               # same val carve as Arm A
    tr = np.array(train_ids, dtype=object)
    rng.shuffle(tr)
    n_val = int(len(tr) * VAL_FRAC)
    val_ids, fit_ids = list(tr[:n_val]), list(tr[n_val:])

    fit_ld = _loader(fit_ids, seqs, demo, label, True)
    val_ld = _loader(val_ids, seqs, demo, label, False)
    test_ld = _loader(list(test_ids_), seqs, demo, label, False)

    model = SeqCNN(vocab_size, len(DEMO_COLS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Headline runs had no schedule at all (COSINE=False keeps that exactly). Run 2 adds
    # cosine decay over the planned budget so both sequence arms get the same treatment.
    steps_per_epoch = max(1, (len(fit_ids) + BATCH - 1) // BATCH)
    total_steps = max(1, steps_per_epoch * MAX_EPOCHS)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: 0.5 * (1.0 + math.cos(math.pi * min(1.0, st / total_steps)))
        if COSINE else 1.0)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, best_epoch, waited = -1.0, None, -1, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, lens, d, y in fit_ld:
            x, lens, d, y = x.to(DEVICE), lens.to(DEVICE), d.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x, lens, d), y).backward()
            opt.step(); sched.step()
        pv, yv = _predict(model, val_ld)
        auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else 0.5
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, waited = auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    model.load_state_dict(best_state)
    pt, yt = _predict(model, test_ld)
    # Persist per-patient test predictions so paired bootstrap / DeLong intervals on the
    # A-vs-B gap can be computed later without re-training. Row order == test_ids_.
    preds = pd.DataFrame({"PAT_ID": list(test_ids_), "y": yt.astype("int8"),
                          "p": pt.astype("float32")})
    return {
        "auroc": float(roc_auc_score(yt, pt)),
        "auprc": float(average_precision_score(yt, pt)),
        "brier": float(brier_score_loss(yt, pt)),
        "best_epoch": int(best_epoch), "val_auroc": float(best_auc),
        "n_fit": len(fit_ids), "n_val": len(val_ids),
        "pos_rate_train": float(np.mean([label[i] for i in fit_ids])),
    }, preds


def build(disease: str = "arthritis", tag: str = "", sizes=None, n_seeds: int = 3):
    """Train Arm B across the ladder.

    `tag` writes to experiments/<disease>/armB<tag>/ so an ablation never overwrites the
    headline runs; `sizes` restricts the rungs.
    """
    seqs, demo, label, vocab_size = _load(disease)
    sizes = ladder_sizes(disease) if sizes is None else list(sizes)
    test = test_ids(disease)

    records = []
    for size in sizes:
        for k in range(n_seeds):
            run = EXPERIMENTS / disease / f"armB{tag}" / f"size{size}_seed{k}"
            # resume: a run counts as done only if BOTH metrics and predictions landed
            if (run / "metrics.json").exists() and (run / "preds.parquet").exists():
                records.append(json.loads((run / "metrics.json").read_text()))
                continue
            sub = train_subset_ids(disease, size, k)
            t0 = time.time()
            m, preds = _fit_eval(seqs, demo, label, vocab_size, sub, test, seed_for(k))
            m.update(disease=disease, arm="B", size=size, seed=k,
                     vocab_size=vocab_size, seconds=round(time.time() - t0, 1))
            run.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(run / "preds.parquet", index=False)
            (run / "config.json").write_text(json.dumps(
                {"disease": disease, "arm": "B", "size": size, "seed": k,
                 "model": "cnn", "tag": tag, "vocab_size": vocab_size,
                 "hyper": dict(HYPER, max_len=MAX_LEN, seq_suffix=_SEQ_SUFFIX)},
                indent=2), encoding="utf-8")
            (run / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            records.append(m)
            print(f"  size={size:>7,} seed={k} AUROC={m['auroc']:.4f} "
                  f"AUPRC={m['auprc']:.4f} (val {m['val_auroc']:.4f}, "
                  f"ep{m['best_epoch']}, {m['seconds']}s)")

    res = pd.DataFrame(records)
    res.to_csv(REPORTS / f"stage7_{disease}_armB{tag}_results.csv", index=False)
    if not tag:
        # The crossover report/figure describe the headline configuration only.
        _report(disease, res, sizes)
        _figure(disease, res, sizes)
    print(f"[arm_b] {disease}{tag}: {len(records)} runs "
          f"-> reports/stage7_{disease}_armB{tag}_results.csv", flush=True)


def _agg(res):
    return res.groupby("size").agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"), brier_mean=("brier", "mean")).reset_index()


def _armA_curve(disease):
    p = REPORTS / f"stage6_{disease}_armA_results.csv"
    if not p.exists():
        return None
    a = pd.read_csv(p)
    a = a[a.featureset == "rich_full"]
    return a.groupby("size").agg(auroc_mean=("auroc", "mean"),
                                 auroc_sd=("auroc", "std")).reset_index()


def _report(disease, res, sizes):
    g = _agg(res)
    a = _armA_curve(disease)
    L = []; w = L.append
    w(f"# Stage 7 — Arm B (1D CNN) and the A-vs-B crossover: {DISEASES[disease]['label']}")
    w("")
    w("_Ordered code sequences -> 1D CNN (Deepr-style). Same ladder, val split, and fixed "
      "temporal test set as Arm A; identical information, different representation. Mean "
      "± SD over 3 seeds._")
    w("")
    w("## Learning curve (temporal holdout, AUROC mean ± SD)")
    if a is not None:
        w("| size | Arm A (rich GBDT) | Arm B (1D CNN) | Δ (B − A) |")
        w("|---|---|---|---|")
        for s in sizes:
            b = g[g["size"] == s].iloc[0]; aa = a[a["size"] == s]
            if len(aa):
                aa = aa.iloc[0]
                w(f"| {s:,} | {aa.auroc_mean:.4f} ± {aa.auroc_sd:.4f} | "
                  f"{b.auroc_mean:.4f} ± {b.auroc_sd:.4f} | {b.auroc_mean - aa.auroc_mean:+.4f} |")
    else:
        w("| size | Arm B (1D CNN) |"); w("|---|---|")
        for s in sizes:
            b = g[g["size"] == s].iloc[0]
            w(f"| {s:,} | {b.auroc_mean:.4f} ± {b.auroc_sd:.4f} |")
    w("")
    w("## AUPRC / calibration (Brier), Arm B")
    w("| size | AUPRC | Brier |"); w("|---|---|---|")
    for s in sizes:
        b = g[g["size"] == s].iloc[0]
        w(f"| {s:,} | {b.auprc_mean:.4f} | {b.brier_mean:.4f} |")
    w("")
    w("## Reading it")
    if a is not None:
        bmax = g[g["size"] == sizes[-1]].iloc[0]
        amax = a[a["size"] == sizes[-1]]
        if len(amax):
            amax = amax.iloc[0]; d = bmax.auroc_mean - amax.auroc_mean
            trend = ("closing" if d < 0 else "ahead")
            w(f"- At the largest local size ({sizes[-1]:,}): Arm A **{amax.auroc_mean:.3f}** "
              f"vs Arm B **{bmax.auroc_mean:.3f}** (Δ {d:+.3f}). Arm B is {trend}.")
            w("- Whether the curves cross depends on scale beyond the local sample; the "
              "≥1M rungs need the full extraction (the study's #1 gate). The **slope** of "
              "Δ across the ladder is the local signal for whether a crossover is coming.")
    w("")
    w(f"Overlay figure: `reports/figures/stage7_{disease}_crossover.png`. "
      f"Raw table: `reports/stage7_{disease}_armB_results.csv`.")
    (REPORTS / f"stage7_{disease}_armB.md").write_text("\n".join(L), encoding="utf-8")


def _figure(disease, res, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = _agg(res); a = _armA_curve(disease)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    if a is not None:
        ax.errorbar(a["size"], a["auroc_mean"], yerr=a["auroc_sd"], marker="s",
                    capsize=3, color="#1b9e77", label="Arm A — engineered features (GBDT)")
    ax.errorbar(g["size"], g["auroc_mean"], yerr=g["auroc_sd"], marker="o",
                capsize=3, color="#d95f02", label="Arm B — code sequence (1D CNN)")
    ax.set_xscale("log")
    ax.set_xlabel("Training patients (log scale)")
    ax.set_ylabel("AUROC (temporal holdout, 2023)")
    ax.set_title(f"Scale crossover: {DISEASES[disease]['label']}")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / f"stage7_{disease}_crossover.png")
    plt.close(fig)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")


def configure(**kw):
    """Override the frozen config for an ABLATION run (mirrors arm_c.configure).

    Unknown keys raise, so a typo fails loudly instead of silently training the wrong model.
    """
    allowed = {"D_MODEL", "KERNELS", "N_FILTERS", "MLP_HID", "DROPOUT", "BATCH", "LR",
               "MAX_EPOCHS", "PATIENCE", "MAX_LEN", "_SEQ_SUFFIX", "COSINE"}
    g = globals()
    for k, v in kw.items():
        if k not in allowed:
            raise KeyError(f"unknown Arm B config key {k!r}; allowed: {sorted(allowed)}")
        g[k] = v
    g["HYPER"] = dict(d_model=D_MODEL, kernels=list(KERNELS), n_filters=N_FILTERS,
                      mlp_hidden=MLP_HID, dropout=DROPOUT, max_len=MAX_LEN, batch=BATCH,
                      lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, val_frac=VAL_FRAC,
                      optimizer="adam", cosine=COSINE, loss="bce", pos_reweight=False)
    return g["HYPER"]
