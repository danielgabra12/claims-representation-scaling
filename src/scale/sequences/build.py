"""Stage 3.2 — per-patient ordered code sequences (Arm B inputs).

Run: python -m src.scale.sequences.build            # arthritis
     python -m src.scale.sequences.build t2d

Arm B (the sequence model) must see the SAME information as Arm A (see
reports/fairness_contract.md) so any win is representation, not extra data. So the
token vocabulary is exactly Arm A's frozen bag vocabulary — the same 500 DX 3-char
categories, 200 procedure codes, and 95 drug classes — loaded from
<disease>_features_vocab.json. The ONLY difference from Arm A is that here the codes
stay in their real month order instead of being collapsed into counts.

For each cohort patient we take the same baseline window Arm A used
[FIRST_MONTH, index_month), emit one token per claim ordered by month, map it through
the frozen vocab (out-of-vocab -> UNK), and keep the most recent MAX_LEN tokens
(recency matters most and it bounds compute). Demographics (the same 7 Arm A has) are
carried alongside so the sequence model isn't blind to age/sex/region.

Leakage discipline is identical to Stage 3.1: baseline window only, vocab frozen from
the train pool, phenotype codes already barred from that vocab.

Writes data/processed/<disease>_sequences.parquet (PAT_ID, seq list<int>, n_tokens,
demographics, label, split, index_year) and *_sequences_vocab.json (token map).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from src.scale.config import (DISEASES, DATA_PROCESSED, REPORTS,
                              RX_TREATMENT_EXCLUDE_USC, RX_TREATMENT_EXCLUDE_CLASS,
                              mem_limit_gb as _mem_limit_gb)
from src.scale.data.access import connect, tbl

PAD, UNK = 0, 1              # reserved token ids
MAX_LEN = 200               # keep most-recent N tokens (p95 baseline length ~178)


def _lit(v):
    return "'" + v.replace("'", "''") + "'"


def _token_vocab(disease: str):
    """Build the token->id map from Arm A's frozen bag vocabulary (exact parity)."""
    v = json.loads((DATA_PROCESSED / f"{disease}_features_vocab.json").read_text())
    toks = (["DX:" + c for c in v["dx_cats"].values()]
            + ["PX:" + c for c in v["proc_codes"].values()]
            + ["RX:" + c for c in v["rx_class_names"].values()])
    # ids 2..N+1 (0=PAD, 1=UNK); order is stable/deterministic from the vocab file
    return {t: i for i, t in enumerate(toks, start=2)}


def build(disease: str = "arthritis", max_len: int | None = None, suffix: str = ""):
    """Build per-patient token sequences.

    `max_len` overrides the truncation cap and `suffix` writes to a separate file, so the
    context-length ablation (len 512) can be produced without disturbing the len-200
    sequences every committed result was trained on.
    """
    if disease not in DISEASES:
        raise KeyError(f"unknown disease {disease!r}")
    max_len = MAX_LEN if max_len is None else int(max_len)
    tok2id = _token_vocab(disease)
    excl = RX_TREATMENT_EXCLUDE_USC.get(disease, [])
    excl_sql = ", ".join(_lit(u) for u in excl) or "''"
    cls_excl = RX_TREATMENT_EXCLUDE_CLASS.get(disease, [])
    cls_sql = ", ".join(_lit(u) for u in cls_excl) or "''"
    tmp = Path(tempfile.mkdtemp(prefix=f"{disease}_seq_"))
    def _log(msg):
        import time
        print(f"[sequences:{disease}] {time.strftime('%H:%M:%S')} {msg}", flush=True)

    con = connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=3")
    # cap DuckDB below free RAM so the window sort spills to disk instead of racing pandas
    # / the OS for the last GB on this 16GB box (a hard OOM-kill leaves no traceback)
    mem_gb = min(_mem_limit_gb(), 6)
    con.execute(f"SET memory_limit='{mem_gb}GB'")
    con.execute("SET max_temp_directory_size='22GiB'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    _log(f"start (duckdb mem_limit={mem_gb}GB, threads=3)")
    cohort_pq = (DATA_PROCESSED / f"{disease}_cohort.parquet").as_posix()
    try:
        con.execute("CREATE MACRO m_abs(mid) AS ((CAST(mid AS BIGINT) // 100) * 12 + (CAST(mid AS BIGINT) % 100) - 1)")
        con.execute(f"CREATE VIEW cohort AS SELECT * FROM read_parquet('{cohort_pq}')")
        con.execute(f"CREATE VIEW claims AS SELECT * FROM {tbl('claims')}")
        con.execute(f"""CREATE VIEW rxlk AS
            SELECT upper(trim(NDC)) AS ndc, THPTC_CLAS_DESC, upper(USC_NAME) AS usc
            FROM {tbl('rx_lookup')}""")

        # baseline-window claims: prefer the pre-sliced parquet (Stage 3.0), else raw scan.
        # VIEWs (not TABLEs): at 2M scale base_claims is ~162M rows and ev derives another
        # ~162M from it; materializing both simultaneously blows the RAM budget. As views,
        # DuckDB streams straight from parquet into the windowed COPY and only spills the
        # sort's working set to temp.
        sliced = DATA_PROCESSED / f"{disease}_base_claims.parquet"
        if sliced.exists():
            con.execute(f"CREATE VIEW base_claims AS SELECT PAT_ID, CODE_TYPE, code_n, "
                        f"ev_abs FROM read_parquet('{sliced.as_posix()}')")
        else:
            con.execute("""
                CREATE VIEW base_claims AS
                SELECT c.PID AS PAT_ID, c.CODE_TYPE, upper(trim(c.CODE)) AS code_n,
                       m_abs(c.MONTH_ID) AS ev_abs
                FROM claims c JOIN cohort b ON c.PID = b.PAT_ID
                WHERE m_abs(c.MONTH_ID) >= m_abs(b.FIRST_MONTH)
                  AND m_abs(c.MONTH_ID) <  m_abs(b.index_month)""")

        # one token per claim; RX mapped to drug class with the same USC + class
        # treatment exclusion as Arm A (identical information)
        con.execute(f"""
            CREATE VIEW ev AS
            SELECT PAT_ID, ev_abs, 'DX:' || substr(code_n,1,3) AS tok
              FROM base_claims WHERE CODE_TYPE='DX'
            UNION ALL
            SELECT PAT_ID, ev_abs, 'PX:' || code_n FROM base_claims WHERE CODE_TYPE='PROC'
            UNION ALL
            SELECT bc.PAT_ID, bc.ev_abs,
                   CASE WHEN r.usc IN ({excl_sql}) OR r.THPTC_CLAS_DESC IN ({cls_sql})
                             OR r.THPTC_CLAS_DESC IS NULL
                        THEN NULL ELSE 'RX:' || r.THPTC_CLAS_DESC END AS tok
              FROM base_claims bc LEFT JOIN rxlk r ON r.ndc = bc.code_n
             WHERE bc.CODE_TYPE='RX'""")

        con.execute("CREATE TABLE vmap(tok VARCHAR, tid INTEGER)")
        con.executemany("INSERT INTO vmap VALUES (?, ?)", list(tok2id.items()))

        # Per-patient sort + most-recent-MAX_LEN truncation + ordered-list aggregation.
        # DuckDB's window and ordered-list operators DON'T spill to disk, so doing this in
        # one pass over the ~162M-row event table OOMs even at a 6GB cap (fine at the old
        # 10% scale). So we bucket patients by hash(PAT_ID) and aggregate one bucket at a
        # time — each bucket's window sort is ~1/NBUCKETS the size and fits in memory:
        #  (1) stream mapped events (PAT_ID, ev_abs, tid) to a bucket-partitioned parquet
        #      (a pure projection -> low, streaming memory);
        #  (2) aggregate each bucket's partition independently into a per-bucket seq parquet;
        #  (3) assemble the compact per-patient result (~1 row/patient) in pandas.
        NBUCKETS = 32
        mapped_dir = tmp / "mapped"
        _log(f"streaming mapped events -> {NBUCKETS} bucket partitions...")
        con.execute(f"""
            COPY (
                SELECT e.PAT_ID, e.ev_abs, COALESCE(v.tid, {UNK}) AS tid,
                       (hash(e.PAT_ID) % {NBUCKETS})::INTEGER AS bucket
                FROM ev e LEFT JOIN vmap v ON v.tok = e.tok
            ) TO '{mapped_dir.as_posix()}'
              (FORMAT PARQUET, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE)""")
        con.execute("DROP VIEW IF EXISTS ev")
        con.execute("DROP VIEW IF EXISTS base_claims")
        con.execute("DROP TABLE IF EXISTS vmap")

        seqs_dir = tmp / "seqs_parts"
        seqs_dir.mkdir(exist_ok=True)
        _log(f"aggregating {NBUCKETS} buckets...")
        for b in range(NBUCKETS):
            src = (mapped_dir / f"bucket={b}" / "*.parquet").as_posix()
            part = (seqs_dir / f"part_{b:02d}.parquet").as_posix()
            con.execute(f"""
                COPY (
                    WITH ranked AS (
                        SELECT PAT_ID, tid,
                               row_number() OVER (PARTITION BY PAT_ID
                                                  ORDER BY ev_abs DESC, tid DESC) AS rn
                        FROM read_parquet('{src}')
                    )
                    SELECT PAT_ID,
                           list(tid ORDER BY rn DESC) AS seq,   -- chronological
                           count(*)                   AS n_tokens
                    FROM ranked
                    WHERE rn <= {max_len}                        -- keep most-recent max_len
                    GROUP BY PAT_ID
                ) TO '{part}' (FORMAT PARQUET)""")
            if (b + 1) % 8 == 0:
                _log(f"  aggregated {b + 1}/{NBUCKETS} buckets")
        _log("duckdb aggregation done; assembling final table in pandas...")
        _assemble(disease, seqs_dir, cohort_pq, suffix)
        _log("assemble done; sequences parquet written")
        vocab_out = {"disease": disease, "pad": PAD, "unk": UNK, "max_len": max_len,
                     "vocab_size": len(tok2id) + 2, "token_to_id": tok2id}
        (DATA_PROCESSED / f"{disease}_sequences{suffix}_vocab.json").write_text(
            json.dumps(vocab_out, indent=2), encoding="utf-8")
        _report(disease, len(tok2id) + 2, max_len, suffix)
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _assemble(disease, seqs_dir: Path, cohort_pq, suffix: str = ""):
    import pandas as pd

    # seqs_dir holds the compact per-bucket DuckDB results (part_*.parquet): one row per
    # patient with >=1 baseline claim, seq already a list<int> in chronological order
    # truncated to <=MAX_LEN, n_tokens set. read_parquet on the dir unions all parts.
    seqs = pd.read_parquet(seqs_dir).set_index("PAT_ID")

    coh = pd.read_parquet(cohort_pq, columns=["PAT_ID", "DER_SEX", "PAT_REGION",
                          "age_at_index", "index_year", "split", "label"]).set_index("PAT_ID")
    out = coh.copy()
    out["age_at_index"] = out["age_at_index"].astype("int16")
    out["is_female"] = (coh["DER_SEX"] == "F").astype("int8")
    out["is_male"] = (coh["DER_SEX"] == "M").astype("int8")
    for r in ("MW", "NE", "S", "W"):
        out[f"reg_{r.lower()}"] = (coh["PAT_REGION"] == r).astype("int8")
    out = out.drop(columns=["DER_SEX", "PAT_REGION"])
    out = out.join(seqs)
    # patients with no baseline claims -> single UNK token (a valid, non-pad token so
    # attention masking never sees an all-pad row), n_tokens 0
    empty = out["seq"].isna()
    out.loc[empty, "seq"] = out.loc[empty, "seq"].apply(lambda _: [UNK])
    out["seq"] = out["seq"].apply(lambda a: list(a))        # parquet-friendly list<int>
    out["n_tokens"] = out["n_tokens"].fillna(0).astype("int32")
    out.reset_index().to_parquet(
        DATA_PROCESSED / f"{disease}_sequences{suffix}.parquet", index=False)


def _report(disease, vocab_size, max_len=MAX_LEN, suffix=""):
    import numpy as np
    import pandas as pd
    df = pd.read_parquet(DATA_PROCESSED / f"{disease}_sequences{suffix}.parquet",
                         columns=["n_tokens", "label", "split"])
    n = len(df); empt = int((df.n_tokens == 0).sum())
    q = df.n_tokens.quantile([0.5, 0.95]).to_dict()
    L = []; w = L.append
    w(f"# Stage 3.2 — Code sequences (Arm B inputs): {DISEASES[disease]['label']}")
    w("")
    w("_Same frozen vocabulary as Arm A (identical information); tokens kept in real "
      "month order. Baseline window only; vocab frozen from the train pool._")
    w("")
    w(f"- **Patients:** {n:,}. **Vocab size:** {vocab_size:,} (incl. PAD, UNK). "
      f"**Max length:** {max_len} (most-recent tokens kept).")
    w(f"- **Sequence length:** median {q[0.5]:.0f}, 95th pct {q[0.95]:.0f}.")
    w(f"- **No-baseline-claim patients:** {empt:,} (single PAD token; demographics only).")
    w("")
    w(f"Table: `data/processed/{disease}_sequences{suffix}.parquet` (gitignored). "
      f"Token map: `{disease}_sequences{suffix}_vocab.json`.")
    (REPORTS / f"stage3_2_{disease}_sequences{suffix}.md").write_text(
        "\n".join(L), encoding="utf-8")
    print(f"[sequences] {disease}: {n:,} patients, vocab {vocab_size:,}, "
          f"median len {q[0.5]:.0f}, p95 {q[0.95]:.0f}, empty {empt:,}")


if __name__ == "__main__":
    # usage: python -m src.scale.sequences.build [disease] [max_len]
    # a non-default max_len writes to <disease>_sequences_len<N>.parquet, leaving the
    # len-200 files (which every committed result used) untouched.
    _d = sys.argv[1] if len(sys.argv) > 1 else "arthritis"
    _ml = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_LEN
    build(_d, _ml, "" if _ml == MAX_LEN else f"_len{_ml}")
