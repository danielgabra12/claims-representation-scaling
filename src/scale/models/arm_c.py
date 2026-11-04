"""Stage 8 — Arm C: ordered code sequences -> Transformer encoder (self-attention).

Answers the obvious reviewer question head-on: "you compared engineered features against a
2017 CNN (Deepr) — did you try a modern Transformer?" Arm C is a BEHRT-style self-attention
encoder (Li et al. 2020) over the SAME tokens, SAME frozen vocabulary, SAME baseline window,
SAME nested ladder, SAME train-carved validation split, SAME fixed temporal test set, and
SAME 3 seeds as Arm A (GBDT) and Arm B (CNN). Only the representation/learner differs, so the
three-way comparison stays inside the fairness contract.

Model: token embedding + learned positional encoding + a [CLS] token -> N-layer Transformer
encoder (multi-head self-attention) -> [CLS] pooling -> concat the same 7 demographics -> MLP
-> logit. Fixed, documented compute budget (no per-size or per-seed search), matching the
parity rule. Transformers are self-attention over all positions and are GPU-friendly; this is
meant to be run on a GPU (Colab T4), exactly like Arm B.

Writes experiments/<disease>/armC/size{size}_seed{k}/, reports/stage8_<disease>_armC.md, the
results CSV, and a three-arm overlay figure reports/figures/stage8_<disease>_threeway.png.
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
MAX_LEN = 200

# --- fixed compute budget (Arm C side of the fairness contract) --------------
# A small-but-real Transformer encoder: sensible, literature-standard config (BEHRT-scale),
# frozen across every size/seed. A short LR warmup is included because self-attention training
# is unstable without it; that is a fixed part of the config, not per-run tuning.
D_MODEL, NHEAD, NLAYERS, FF, DROPOUT = 128, 8, 2, 256, 0.2
BATCH, LR, WD, MAX_EPOCHS, PATIENCE, VAL_FRAC, WARMUP = 512, 3e-4, 1e-2, 25, 4, 0.10, 300
COSINE = False             # headline runs: warmup only. Run 2 turns this on.
SEQ_SUFFIX = ""            # "" = the len-200 sequences all committed results used


def _hyper():
    """Current config as a dict, read at call time so ablations record what they ran."""
    return dict(d_model=D_MODEL, n_heads=NHEAD, n_layers=NLAYERS, ff=FF, dropout=DROPOUT,
                max_len=MAX_LEN, batch=BATCH, lr=LR, weight_decay=WD, warmup_steps=WARMUP,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, val_frac=VAL_FRAC,
                optimizer="adamw", cosine=COSINE, loss="bce", pos_reweight=False, pooling="cls",
                seq_suffix=SEQ_SUFFIX)


HYPER = _hyper()


def configure(**kw):
    """Override the frozen config for an ABLATION run.

    The main result is one fixed configuration (the fairness contract). The ablations
    asked for by the audit (does capacity change the conclusion? does a longer context
    change it?) need to vary exactly one axis at a time and write somewhere else, so this
    mutates the module globals and the caller passes a distinct `tag` to build().
    Accepted keys are the names below; anything else is a typo and raises.
    """
    allowed = {"D_MODEL", "NHEAD", "NLAYERS", "FF", "DROPOUT", "BATCH", "LR", "WD",
               "MAX_EPOCHS", "PATIENCE", "WARMUP", "MAX_LEN", "SEQ_SUFFIX", "COSINE"}
    g = globals()
    for k, v in kw.items():
        if k not in allowed:
            raise KeyError(f"unknown Arm C config key {k!r}; allowed: {sorted(allowed)}")
        g[k] = v
    g["HYPER"] = _hyper()
    return _hyper()

torch.set_num_threads(8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class SeqTransformer(nn.Module):
    """BEHRT-style Transformer encoder over the code sequence (+ [CLS] + positional emb)."""
    def __init__(self, vocab_size, n_demo):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN + 1, D_MODEL)               # +1 for the CLS position
        self.cls = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, FF, DROPOUT,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, NLAYERS)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL + n_demo, MLP := 128), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(MLP, 1))

    def forward(self, x, lens, demo):
        B, L = x.shape
        cls = self.cls.expand(B, 1, D_MODEL)
        h = torch.cat([cls, self.emb(x)], dim=1)                   # (B, L+1, D)
        pos = torch.arange(L + 1, device=x.device).clamp(max=MAX_LEN)[None, :]
        h = h + self.pos(pos)
        padx = torch.arange(L, device=x.device)[None, :] >= lens[:, None]   # (B, L) True=pad
        clsm = torch.zeros(B, 1, dtype=torch.bool, device=x.device)         # CLS never masked
        mask = torch.cat([clsm, padx], dim=1)                      # (B, L+1)
        h = self.enc(h, src_key_padding_mask=mask)
        pooled = h[:, 0]                                            # CLS representation
        return self.head(torch.cat([pooled, demo], dim=1)).squeeze(1)


def _load(disease: str):
    df = pd.read_parquet(
        DATA_PROCESSED / f"{disease}_sequences{SEQ_SUFFIX}.parquet").set_index("PAT_ID")
    vocab = json.loads(
        (DATA_PROCESSED / f"{disease}_sequences{SEQ_SUFFIX}_vocab.json").read_text())
    seqs = {i: np.asarray(s, dtype=np.int64) for i, s in df["seq"].items()}
    demo_mat = df[["age_at_index", "is_female", "is_male",
                   "reg_mw", "reg_ne", "reg_s", "reg_w"]].to_numpy(dtype=np.float32).copy()
    demo_mat[:, 0] = demo_mat[:, 0] / 100.0
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
        ps.append(torch.sigmoid(model(x, lens, demo)).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def _fit_eval(seqs, demo, label, vocab_size, train_ids, test_ids_, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)               # same val carve rule as Arm A / Arm B
    tr = np.array(train_ids, dtype=object)
    rng.shuffle(tr)
    n_val = int(len(tr) * VAL_FRAC)
    val_ids, fit_ids = list(tr[:n_val]), list(tr[n_val:])

    fit_ld = _loader(fit_ids, seqs, demo, label, True)
    val_ld = _loader(val_ids, seqs, demo, label, False)
    test_ld = _loader(list(test_ids_), seqs, demo, label, False)

    model = SeqTransformer(vocab_size, len(DEMO_COLS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    # Warmup, then optionally cosine decay to zero over the full planned schedule. The
    # headline runs used warmup only (COSINE=False), so the default path is unchanged; the
    # longer-schedule ablation turns decay on. total_steps is the planned budget, not the
    # early-stopped length, so the shape does not depend on when stopping happens.
    steps_per_epoch = max(1, (len(fit_ids) + BATCH - 1) // BATCH)
    total_steps = max(WARMUP + 1, steps_per_epoch * MAX_EPOCHS)

    def _lr(step):
        if step < WARMUP:
            return (step + 1) / WARMUP
        if not COSINE:
            return 1.0
        prog = min(1.0, (step - WARMUP) / max(1, total_steps - WARMUP))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, best_epoch, waited = -1.0, None, -1, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, lens, d, y in fit_ld:
            x, lens, d, y = x.to(DEVICE), lens.to(DEVICE), d.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x, lens, d), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    # Per-patient test predictions are persisted alongside metrics so that paired
    # bootstrap / DeLong intervals on the A-vs-B-vs-C gaps can be computed after the
    # fact without re-training. Row order matches test_ids_ (loader is shuffle=False).
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
    """Train Arm C across the ladder.

    `tag` writes to experiments/<disease>/armC<tag>/ so an ablation never overwrites the
    headline runs; `sizes` restricts the rungs (ablations only need the informative ones).
    """
    seqs, demo, label, vocab_size = _load(disease)
    sizes = ladder_sizes(disease) if sizes is None else list(sizes)
    test = test_ids(disease)

    records = []
    for size in sizes:
        for k in range(n_seeds):
            run = EXPERIMENTS / disease / f"armC{tag}" / f"size{size}_seed{k}"
            # resume: a run counts as done only if BOTH metrics and predictions landed
            if (run / "metrics.json").exists() and (run / "preds.parquet").exists():
                records.append(json.loads((run / "metrics.json").read_text()))
                continue
            sub = train_subset_ids(disease, size, k)
            t0 = time.time()
            m, preds = _fit_eval(seqs, demo, label, vocab_size, sub, test, seed_for(k))
            m.update(disease=disease, arm="C", size=size, seed=k,
                     vocab_size=vocab_size, seconds=round(time.time() - t0, 1))
            run.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(run / "preds.parquet", index=False)
            (run / "config.json").write_text(json.dumps(
                {"disease": disease, "arm": "C", "size": size, "seed": k,
                 "model": "transformer", "tag": tag, "hyper": _hyper(),
                 "vocab_size": vocab_size},
                indent=2), encoding="utf-8")
            (run / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
            records.append(m)
            print(f"  size={size:>7,} seed={k} AUROC={m['auroc']:.4f} "
                  f"AUPRC={m['auprc']:.4f} (val {m['val_auroc']:.4f}, "
                  f"ep{m['best_epoch']}, {m['seconds']}s)", flush=True)

    res = pd.DataFrame(records)
    res.to_csv(REPORTS / f"stage8_{disease}_armC{tag}_results.csv", index=False)
    if not tag:
        # The three-way report/figure describe the headline configuration only; an
        # ablation just drops its results CSV and is written up alongside the others.
        _report(disease, res, sizes)
        _figure(disease, res, sizes)
    print(f"[arm_c] {disease}{tag}: {len(records)} runs "
          f"-> reports/stage8_{disease}_armC{tag}_results.csv", flush=True)


def _agg(res):
    return res.groupby("size").agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                   auprc_mean=("auprc", "mean")).reset_index()


def _armA_curve(disease):
    p = REPORTS / f"stage6_{disease}_armA_results.csv"
    if not p.exists():
        return None
    a = pd.read_csv(p); a = a[a.featureset == "rich_full"]
    return a.groupby("size").auroc.agg(["mean", "std"]).rename(columns={"mean": "m", "std": "s"})


def _armB_curve(disease):
    p = REPORTS / f"stage7_{disease}_armB_results.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).groupby("size").auroc.agg(["mean", "std"]).rename(columns={"mean": "m", "std": "s"})


def _report(disease, res, sizes):
    g = _agg(res); a = _armA_curve(disease); b = _armB_curve(disease)
    L = []; w = L.append
    w(f"# Stage 8 — Arm C (Transformer) three-way comparison: {DISEASES[disease]['label']}")
    w("")
    w("_Self-attention encoder (BEHRT-style) over the same tokens/vocab/ladder/val/test/seeds "
      "as Arm A (GBDT) and Arm B (CNN). Fixed documented config. Mean ± SD over 3 seeds._")
    w("")
    w("## Learning curves (temporal holdout, AUROC mean ± SD)")
    head = "| size | Arm A (GBDT) | Arm B (CNN) | Arm C (Transformer) |"
    w(head); w("|" + "---|" * 4)
    for s in sizes:
        cc = g[g["size"] == s].iloc[0]
        av = f"{a.loc[s,'m']:.4f}" if a is not None and s in a.index else "—"
        bv = f"{b.loc[s,'m']:.4f}" if b is not None and s in b.index else "—"
        w(f"| {s:,} | {av} | {bv} | {cc.auroc_mean:.4f} ± {cc.auroc_sd:.4f} |")
    w("")
    # Guard the same way the table loop does: a missing top rung in the Arm A/B curves
    # must not raise here, after every model has already been trained.
    smax = sizes[-1]
    if (a is not None and smax in a.index) and (b is not None and smax in b.index):
        av = a.loc[smax, "m"]; bv = b.loc[smax, "m"]; cv = g[g["size"] == smax].iloc[0].auroc_mean
        w("## Reading it")
        w(f"- At the largest local size ({smax:,}): Arm A **{av:.3f}**, Arm B (CNN) **{bv:.3f}**, "
          f"Arm C (Transformer) **{cv:.3f}**.")
        # State the comparison across the WHOLE ladder, not just the top rung: on this
        # data the Transformer loses to the CNN at the smallest size, so a top-rung-only
        # verdict ("beats the 2017 CNN") would be false as a general claim.
        wins = [s for s in sizes if (b is not None and s in b.index
                                     and g[g["size"] == s].iloc[0].auroc_mean > b.loc[s, "m"])]
        losses = [s for s in sizes if s not in wins]
        verdict = ("beats" if cv > bv else "does not beat")
        w(f"- At this rung the Transformer **{verdict}** the 2017 CNN (Δ {cv-bv:+.3f}); vs "
          f"engineered features the gap is {cv-av:+.3f}.")
        if losses and wins:
            w(f"- **Not uniform across scale:** the Transformer is behind the CNN at "
              f"{', '.join(f'{s:,}' for s in losses)} and ahead at "
              f"{', '.join(f'{s:,}' for s in wins)}. Report the crossover, not a blanket "
              "'modern beats old' claim.")
    w("")
    w(f"Overlay: `reports/figures/stage8_{disease}_threeway.png`. Raw: `reports/stage8_{disease}_armC_results.csv`.")
    (REPORTS / f"stage8_{disease}_armC.md").write_text("\n".join(L), encoding="utf-8")


def _figure(disease, res, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = _agg(res); a = _armA_curve(disease); b = _armB_curve(disease)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)
    if a is not None:
        ax.errorbar(a.index, a["m"], yerr=a["s"], marker="s", capsize=3, lw=2.4,
                    color="#33414B", label="Arm A — engineered features (GBDT)")
    if b is not None:
        ax.errorbar(b.index, b["m"], yerr=b["s"], marker="^", capsize=3, lw=2.4,
                    color="#C89F5D", label="Arm B — code sequence (CNN, 2017)")
    ax.errorbar(g["size"], g["auroc_mean"], yerr=g["auroc_sd"], marker="o", capsize=3, lw=2.6,
                color="#A6192E", label="Arm C — code sequence (Transformer)")
    ax.set_xscale("log")
    ax.set_xlabel("Training patients (log scale)")
    ax.set_ylabel("AUROC (temporal holdout, 2023)")
    ax.set_title(f"Three-way scaling comparison: {DISEASES[disease]['label']}")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / f"stage8_{disease}_threeway.png")
    plt.close(fig)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")
