"""Stage 5 — nested patient-level scaling ladder.

Run: python -m src.scale.scaling.ladder            # arthritis

Builds, for the TRAIN pool of a disease, nested random subsets at each ladder size
and each seed. Invariants (spec Section 6, enforced by assertions):
  - patient-level: subsets are sets of PAT_IDs; a patient is train xor test.
  - nested: subset(size_i, seed) is a strict subset of subset(size_{i+1}, seed).
  - fixed test: the temporal test set is defined once (Stage 2) and never grows.
  - >=3 seeds per size: both the subsample draw and (later) model init vary by seed.

Storage: per seed we persist the full shuffled train order once
(data/splits/<disease>_ladder.parquet, columns seed/rank/PAT_ID); subset(size,seed)
= the first `size` ranks. This is compact and makes nesting exact by construction.

Public API used by Stages 6/7:
  ladder_sizes(disease) -> [int, ...]           actual sizes (capped at train pool)
  train_subset_ids(disease, size, seed) -> list[str]
  test_ids(disease) -> list[str]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.scale.config import (DISEASES, DATA_PROCESSED, DATA_SPLITS, REPORTS,
                              SCALING_LADDER, N_SEEDS_PER_POINT, seed_for)
from src.scale.data.access import connect


def _paths(disease: str):
    return (DATA_SPLITS / f"{disease}_ladder.parquet",
            DATA_SPLITS / f"{disease}_ladder_meta.json")


def _load_pools(disease: str):
    cohort_pq = (DATA_PROCESSED / f"{disease}_cohort.parquet").as_posix()
    con = connect()
    df = con.execute(
        f"SELECT PAT_ID, split FROM read_parquet('{cohort_pq}')").df()
    train = df.loc[df.split == "train", "PAT_ID"].to_numpy()
    test = df.loc[df.split == "test", "PAT_ID"].to_numpy()
    return train, test


def _sizes(n_train: int) -> list[int]:
    sizes = [s for s in SCALING_LADDER if s < n_train]
    sizes.append(n_train)          # the "max available" rung
    return sorted(set(sizes))


def build(disease: str = "arthritis"):
    if disease not in DISEASES:
        raise KeyError(f"unknown disease {disease!r}")
    train, test = _load_pools(disease)
    n_train, n_test = len(train), len(test)
    sizes = _sizes(n_train)

    # persist per-seed shuffled train order
    rows = []
    for k in range(N_SEEDS_PER_POINT):
        rng = np.random.default_rng(seed_for(k))
        order = rng.permutation(train)
        rows.append(pd.DataFrame({"seed": k, "rank": np.arange(len(order)),
                                  "PAT_ID": order}))
    ladder = pd.concat(rows, ignore_index=True)
    lpath, mpath = _paths(disease)
    ladder.to_parquet(lpath, index=False)

    # test set persisted once
    pd.DataFrame({"PAT_ID": test}).to_parquet(
        DATA_SPLITS / f"{disease}_test.parquet", index=False)

    meta = {"disease": disease, "n_train_pool": int(n_train), "n_test": int(n_test),
            "sizes": sizes, "n_seeds": N_SEEDS_PER_POINT,
            "requested_ladder": SCALING_LADDER,
            "unreachable_locally": [s for s in SCALING_LADDER if s >= n_train]}
    mpath.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _assert_invariants(disease, sizes, test)
    _report(disease, meta)
    print(f"[ladder] {disease}: train pool {n_train:,}, test {n_test:,}, "
          f"sizes {sizes}, {N_SEEDS_PER_POINT} seeds. "
          f"Unreachable locally: {meta['unreachable_locally']}")


def ladder_sizes(disease: str) -> list[int]:
    _, mpath = _paths(disease)
    return json.loads(mpath.read_text())["sizes"]


def train_subset_ids(disease: str, size: int, seed: int) -> list[str]:
    lpath, _ = _paths(disease)
    con = connect()
    return con.execute(
        f"SELECT PAT_ID FROM read_parquet('{lpath.as_posix()}') "
        f"WHERE seed={int(seed)} AND rank < {int(size)} ORDER BY rank").df()["PAT_ID"].tolist()


def test_ids(disease: str) -> list[str]:
    con = connect()
    return con.execute(
        f"SELECT PAT_ID FROM read_parquet('{(DATA_SPLITS / f'{disease}_test.parquet').as_posix()}')"
    ).df()["PAT_ID"].tolist()


def _assert_invariants(disease, sizes, test):
    test_set = set(test)
    for k in range(N_SEEDS_PER_POINT):
        prev = None
        for s in sizes:
            ids = set(train_subset_ids(disease, s, k))
            assert len(ids) == s, f"size mismatch {len(ids)}!={s}"
            assert ids.isdisjoint(test_set), "train subset overlaps test set"
            if prev is not None:
                assert prev.issubset(ids), "nesting violated"
            prev = ids


def _report(disease, meta):
    L = []; w = L.append
    w(f"# Stage 5 — Scaling ladder: {DISEASES[disease]['label']}")
    w("")
    w("_Nested patient-level train subsets; fixed temporal test set. Invariants "
      "(nesting, size, test-disjointness) asserted for every (size, seed)._")
    w("")
    w(f"- **Train pool:** {meta['n_train_pool']:,} patients. **Fixed test set:** "
      f"{meta['n_test']:,} (never grows).")
    w(f"- **Seeds per size:** {meta['n_seeds']}.")
    w(f"- **Ladder sizes (local):** {meta['sizes']}.")
    w("")
    w("## Scale ceiling (honest gate)")
    if meta["unreachable_locally"]:
        w(f"- Requested rungs **{meta['unreachable_locally']}** exceed the local train "
          f"pool ({meta['n_train_pool']:,}) and are **not reachable on this sample**. "
          "The headline >=1M comparison requires the full-population extraction "
          "(~10x this sample). This is the crossover study's #1 gate, surfaced here.")
    else:
        w("- All requested rungs reachable locally.")
    w("")
    w("Membership: `data/splits/<disease>_ladder.parquet` (seed/rank/PAT_ID); "
      "test: `data/splits/<disease>_test.parquet`.")
    (REPORTS / f"stage5_{disease}_ladder.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")
