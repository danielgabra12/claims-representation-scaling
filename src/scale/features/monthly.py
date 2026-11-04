"""Arm A-M inputs: the monthly-resolution bag (the information-equivalence control).

Run: ./surf/Scripts/python.exe -m src.scale.features.monthly arthritis

Why this exists. Claims are month granular and within-month order is arbitrary, so the ONLY
information the sequence arms (B, C) hold that the order-free bag (Arm A) lacks is which of the
12 observation months an event fell in. Arm A-M gives the bag exactly that, and nothing else:
every count feature is split into 12 monthly bins. If Arm A-M closes the gap to the sequence
arms, then what those arms buy is temporal resolution rather than learned representation, and
the paper's conclusion changes.

Design, kept minimal so the only delta versus Arm A is binning:
  BINNED (x12)   : 500 dx categories, 200 procedure codes, 96 therapeutic classes,
                   8 utilisation counts                      -> 804 * 12 = 9,648 columns
  NOT BINNED     : 7 demographics, 12 comorbidity flags, 5 recency/trend  ->    24 columns
                   (recency is redundant once binned; it is kept anyway so that binning is
                    the sole difference from Arm A)
  Total 9,672 columns. Same frozen vocabulary, same treatment exclusions, same cohort,
  same test set.

Storage. Dense would be 1.25M x 9,672 int16 = ~24 GB, which fits nowhere available. The matrix
is ~99.6% zero, so it is stored as CSR (scipy .npz) with a parallel PAT_ID row index. XGBoost
consumes CSR directly.

Column layout (stable, recorded in the meta file):
    0..23                      unbinned block, in the order given by meta["unbinned"]
    24 + slot*12 + m           code slot (dx 0-499, proc 500-699, rx 700-795), month m in 0..11
    24 + 796*12 + u*12 + m     utilisation feature u in 0..7, month m in 0..11
Month m is the offset from the patient's first observation month, so m=0 is the earliest
observed month and m=11 is the month immediately before index.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from src.scale.config import (DATA_PROCESSED, DISEASES, REPORTS,
                              RX_TREATMENT_EXCLUDE_CLASS, RX_TREATMENT_EXCLUDE_USC,
                              mem_limit_gb as _mem_limit_gb)
from src.scale.data.access import connect, tbl
from src.scale.features.engineered import (COMORBIDITIES, DEMO_FEATURES,
                                           TEMPORAL_FEATURES, UTILIZATION_FEATURES)

N_MONTHS = 12
UNBINNED = DEMO_FEATURES + list(COMORBIDITIES) + TEMPORAL_FEATURES   # 7 + 12 + 5 = 24


def _lit(v):
    return "'" + v.replace("'", "''") + "'"


def _slots(disease: str):
    """Stable code-slot assignment from Arm A's frozen vocabulary (exact parity)."""
    v = json.loads((DATA_PROCESSED / f"{disease}_features_vocab.json").read_text())
    dx = list(v["dx_cats"].values())
    px = list(v["proc_codes"].values())
    rx = list(v["rx_class_names"].values())
    return dx, px, rx


def build(disease: str = "arthritis"):
    if disease not in DISEASES:
        raise KeyError(f"unknown disease {disease!r}")
    import pandas as pd
    from scipy import sparse

    dx, px, rx = _slots(disease)
    n_code_slots = len(dx) + len(px) + len(rx)
    off_codes = len(UNBINNED)
    off_util = off_codes + n_code_slots * N_MONTHS
    n_cols = off_util + len(UTILIZATION_FEATURES) * N_MONTHS
    print(f"[monthly] {n_code_slots} code slots x {N_MONTHS} months + "
          f"{len(UTILIZATION_FEATURES)} utilisation x {N_MONTHS} + {len(UNBINNED)} unbinned "
          f"= {n_cols:,} columns", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix=f"{disease}_monthly_"))
    con = connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=2")
    con.execute(f"SET memory_limit='{min(_mem_limit_gb(), 6)}GB'")
    con.execute("SET max_temp_directory_size='22GiB'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    try:
        # ---- row index: cohort order, identical to the Arm A feature matrix ----------
        feats_pq = (DATA_PROCESSED / f"{disease}_features.parquet").as_posix()
        con.execute(f"""CREATE TABLE pidx AS
            SELECT PAT_ID, (row_number() OVER (ORDER BY PAT_ID) - 1)::INTEGER AS r
            FROM read_parquet('{feats_pq}')""")
        n_rows = con.execute("SELECT COUNT(*) FROM pidx").fetchone()[0]
        print(f"[monthly] {n_rows:,} patients", flush=True)

        # ---- code slot lookup --------------------------------------------------------
        con.execute("CREATE TABLE slot(kind VARCHAR, code VARCHAR, s INTEGER)")
        con.executemany("INSERT INTO slot VALUES (?,?,?)",
                        [("DX", c, i) for i, c in enumerate(dx)]
                        + [("PROC", c, len(dx) + i) for i, c in enumerate(px)]
                        + [("RX", c, len(dx) + len(px) + i) for i, c in enumerate(rx)])

        excl = ", ".join(_lit(u) for u in RX_TREATMENT_EXCLUDE_USC.get(disease, [])) or "''"
        cls = ", ".join(_lit(u) for u in RX_TREATMENT_EXCLUDE_CLASS.get(disease, [])) or "''"
        bc = (DATA_PROCESSED / f"{disease}_base_claims.parquet").as_posix()
        con.execute(f"""CREATE VIEW rxlk AS
            SELECT upper(trim(NDC)) AS ndc, THPTC_CLAS_DESC, upper(USC_NAME) AS usc
            FROM {tbl('rx_lookup')}""")
        # m = months since the patient's first observed month, 0..11
        con.execute(f"""CREATE VIEW bc AS
            SELECT PAT_ID, CODE_TYPE, code_n, (ev_abs - index_abs + {N_MONTHS}) AS m
            FROM read_parquet('{bc}')""")

        # ---- build in ROW BLOCKS -----------------------------------------------------
        # A single global sort over ~90M (row, col) pairs exceeds the memory budget on this
        # machine (DuckDB OOMs at 2.7 GiB). Row blocks are contiguous ranges of the cohort
        # ordering, so each block's CSR can be built independently and the blocks stacked in
        # order: nothing ever sorts or materialises the whole matrix.
        N_BLOCKS = 16
        bsize = (n_rows + N_BLOCKS - 1) // N_BLOCKS
        print(f"[monthly] building in {N_BLOCKS} row blocks of {bsize:,}...", flush=True)

        # unbinned block first (straight from the Arm A matrix, so parity is exact)
        unb = pd.read_parquet(feats_pq, columns=["PAT_ID"] + UNBINNED).sort_values("PAT_ID")
        pat_order = unb["PAT_ID"].to_numpy()
        U = unb[UNBINNED].to_numpy(dtype=np.float32)
        del unb

        blocks, nnz = [], 0
        for b in range(N_BLOCKS):
            lo, hi = b * bsize, min((b + 1) * bsize, n_rows)
            t0 = time.time()
            df = con.execute(f"""
                WITH pb AS (SELECT PAT_ID, r FROM pidx WHERE r >= {lo} AND r < {hi}),
                     bb AS (SELECT b.* FROM bc b SEMI JOIN pb ON pb.PAT_ID = b.PAT_ID),
                     ev AS (
                        SELECT x.PAT_ID, s.s AS slot, x.m
                          FROM bb x JOIN slot s
                            ON s.kind='DX' AND s.code = substr(x.code_n,1,3)
                         WHERE x.CODE_TYPE='DX'
                        UNION ALL
                        SELECT x.PAT_ID, s.s, x.m
                          FROM bb x JOIN slot s ON s.kind='PROC' AND s.code = x.code_n
                         WHERE x.CODE_TYPE='PROC'
                        UNION ALL
                        SELECT x.PAT_ID, s.s, x.m
                          FROM bb x
                          JOIN rxlk rl ON rl.ndc = x.code_n
                          JOIN slot s ON s.kind='RX' AND s.code = rl.THPTC_CLAS_DESC
                         WHERE x.CODE_TYPE='RX'
                           AND rl.usc NOT IN ({excl})
                           AND rl.THPTC_CLAS_DESC NOT IN ({cls})
                     ),
                     codecnt AS (
                        SELECT PAT_ID, {off_codes} + slot*{N_MONTHS} + m AS c, COUNT(*) AS v
                        FROM ev GROUP BY 1,2
                     ),
                     ucnt AS (
                        SELECT PAT_ID, m,
                               COUNT(*)                                          AS u0,
                               SUM(CASE WHEN CODE_TYPE='DX'   THEN 1 ELSE 0 END) AS u1,
                               SUM(CASE WHEN CODE_TYPE='PROC' THEN 1 ELSE 0 END) AS u2,
                               SUM(CASE WHEN CODE_TYPE='RX'   THEN 1 ELSE 0 END) AS u3,
                               COUNT(DISTINCT CASE WHEN CODE_TYPE='DX'   THEN code_n END) AS u4,
                               COUNT(DISTINCT CASE WHEN CODE_TYPE='RX'   THEN code_n END) AS u5,
                               COUNT(DISTINCT CASE WHEN CODE_TYPE='PROC' THEN code_n END) AS u6,
                               1                                                 AS u7
                        FROM bb GROUP BY 1,2
                     ),
                     ulong AS (
                        SELECT PAT_ID, {off_util} + u*{N_MONTHS} + m AS c, v FROM (
                            SELECT PAT_ID, m, 0 u, u0 v FROM ucnt UNION ALL
                            SELECT PAT_ID, m, 1, u1 FROM ucnt UNION ALL
                            SELECT PAT_ID, m, 2, u2 FROM ucnt UNION ALL
                            SELECT PAT_ID, m, 3, u3 FROM ucnt UNION ALL
                            SELECT PAT_ID, m, 4, u4 FROM ucnt UNION ALL
                            SELECT PAT_ID, m, 5, u5 FROM ucnt UNION ALL
                            SELECT PAT_ID, m, 6, u6 FROM ucnt UNION ALL
                            SELECT PAT_ID, m, 7, u7 FROM ucnt
                        ) WHERE v > 0
                     )
                SELECT (p.r - {lo})::INTEGER AS r, x.c::INTEGER AS c, x.v::FLOAT AS v
                FROM (SELECT * FROM codecnt UNION ALL SELECT * FROM ulong) x
                JOIN pb p ON p.PAT_ID = x.PAT_ID
            """).df()
            rr = df["r"].to_numpy(np.int32)
            cc = df["c"].to_numpy(np.int32)
            vv = df["v"].to_numpy(np.float32)
            del df
            blk = sparse.coo_matrix((vv, (rr, cc)), shape=(hi - lo, n_cols)).tocsr()
            del rr, cc, vv
            blk[:, :len(UNBINNED)] = U[lo:hi]        # splice in the unbinned columns
            blk.eliminate_zeros()
            nnz += blk.nnz
            blocks.append(blk)
            print(f"  block {b+1}/{N_BLOCKS} rows {lo:,}-{hi:,} nnz={blk.nnz:,} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        print("[monthly] stacking blocks...", flush=True)
        X = sparse.vstack(blocks, format="csr")
        del blocks
        dens = 100.0 * X.nnz / (n_rows * n_cols)
        print(f"[monthly] {X.nnz:,} non-zeros ({dens:.3f}% dense)", flush=True)
        X.sort_indices()
        print(f"[monthly] final matrix {X.shape} nnz={X.nnz:,} "
              f"({X.data.nbytes/1e6:.0f} MB data)", flush=True)

        out = DATA_PROCESSED / f"{disease}_monthly.npz"
        sparse.save_npz(out, X)
        np.save(DATA_PROCESSED / f"{disease}_monthly_rows.npy", pat_order)
        meta = {"disease": disease, "n_rows": int(n_rows), "n_cols": int(n_cols),
                "nnz": int(X.nnz), "n_months": N_MONTHS,
                "unbinned": UNBINNED, "off_codes": off_codes, "off_util": off_util,
                "n_code_slots": n_code_slots,
                "utilization": UTILIZATION_FEATURES,
                "dx_k": len(dx), "proc_k": len(px), "rx_k": len(rx)}
        (DATA_PROCESSED / f"{disease}_monthly_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        _report(disease, meta, dens)
        print(f"[monthly] wrote {out}", flush=True)
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _report(disease, meta, dens):
    L = []
    w = L.append
    w(f"# Arm A-M inputs: monthly-resolution bag ({DISEASES[disease]['label']})")
    w("")
    w("_The information-equivalence control. Claims are month granular, so the only thing the "
      "sequence arms hold that the order-free bag lacks is which of the 12 observation months "
      "an event fell in. Arm A-M gives the bag exactly that and nothing else._")
    w("")
    w(f"- **Columns:** {meta['n_cols']:,} "
      f"({meta['n_code_slots']} code slots + {len(meta['utilization'])} utilisation counts, "
      f"each x {meta['n_months']} months, plus {len(meta['unbinned'])} unbinned).")
    w(f"- **Rows:** {meta['n_rows']:,}. **Non-zeros:** {meta['nnz']:,} "
      f"(**{dens:.2f}%** dense).")
    w(f"- Dense storage would be "
      f"{meta['n_rows']*meta['n_cols']*2/1e9:.0f} GB, so the matrix is stored as CSR.")
    w("")
    w("**Not binned** (so that binning is the only difference from Arm A): "
      + ", ".join(f"`{c}`" for c in meta["unbinned"]))
    w("")
    w("Same frozen vocabulary, same treatment exclusions, same cohort and test set as Arm A.")
    (REPORTS / f"stage9_{disease}_monthly.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")
