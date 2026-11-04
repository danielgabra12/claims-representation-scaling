"""Stage 3.1 — enriched engineered per-patient features (Arm A inputs).

Run: python -m src.scale.features.engineered            # arthritis
     python -m src.scale.features.engineered t2d

This is the fair-baseline upgrade of the original 27-feature Arm A. Rationale and the
rules it obeys live in reports/fairness_contract.md. In short: Arm B sees the raw code
sequence, so Arm A must see the same code CONTENT (as order-free bags), or a loss to
Arm B could just mean Arm A was starved. Feature families:

  demo/comorbidity/utilization  : as before (the original 27 survive as `thin`).
  dx_bag                        : count per 3-char ICD-10 category (top-K by train-pool
                                  patient frequency, label-free, frozen).
  rx_class                      : count per therapeutic class (rx_lookup.THPTC_CLAS_DESC),
                                  disease-specific DMARDs/biologics/urate-lowering removed
                                  (config.RX_TREATMENT_EXCLUDE_USC, decision #1).
  proc_bag                      : count per procedure code (top-K by train-pool freq).
  temporal                      : light recency/trend (5 features).

Leakage discipline (audited below):
  - every feature from the baseline window [FIRST_MONTH, index) only;
  - vocabularies chosen from the TRAIN POOL only (split='train'), by patient frequency
    (never labels), then frozen and applied to all patients incl. the test set;
  - phenotype codes cannot enter dx_bag (cohort is disease-free at index; a defensive
    regex filter drops them from the vocab regardless).

Memory/disk discipline: the machine has ~29 GB free on a single drive and claims is 290M
rows, so a wide in-memory aggregation OOMs / fills the disk. Everything is therefore
streamed through on-disk parquet: base_claims and each per-family aggregate are written to
a scratch dir, dropped from memory, and the final wide table is assembled by joining those
parquets with a streaming COPY.

Writes data/processed/<disease>_features.parquet, *_features_meta.json (family tags +
the `full`/`clinical`/`thin` column sets Stage 6 selects on) and *_features_vocab.json
(the frozen vocabulary, for reproducibility).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from src.scale.config import (DISEASES, DATA_PROCESSED, REPORTS,
                              RICH_DX_VOCAB_K, RICH_PROC_VOCAB_K,
                              RX_TREATMENT_EXCLUDE_USC, RX_TREATMENT_EXCLUDE_CLASS,
                              mem_limit_gb as _mem_limit_gb)
from src.scale.data.access import connect, tbl

# Curated comorbidity groups (ICD-10, no dots). Deliberately NOT the arthritis
# phenotype codes -> they predict onset without leaking the label. Documented,
# defensible, arthritis-relevant (Elixhauser-inspired + MSK symptom codes).
COMORBIDITIES = {
    "cm_htn":          r"^I1[0-6]",        # hypertension
    "cm_lipid":        r"^E78",            # hyperlipidemia
    "cm_diabetes":     r"^E0[89]|^E1[0-3]",# diabetes mellitus
    "cm_obesity":      r"^E66",            # obesity
    "cm_depr_anx":     r"^F3[23]|^F41",    # depression / anxiety
    "cm_ckd":          r"^N18",            # chronic kidney disease
    "cm_ihd":          r"^I2[0-5]",        # ischemic heart disease
    "cm_osteoporosis": r"^M8[01]",         # osteoporosis
    "cm_backpain":     r"^M54",            # dorsalgia / back pain
    "cm_jointpain":    r"^M25",            # joint pain / arthralgia (symptom, pre-dx)
    "cm_soft_msk":     r"^M7[0-9]",        # other soft-tissue / MSK (incl. fibromyalgia)
    "cm_tobacco":      r"^F17|^Z87891",    # tobacco use / history
}

# Utilization = pure volume/intensity. Removed in the access-confound ablation.
UTILIZATION_FEATURES = [
    "n_claims", "n_dx", "n_proc", "n_rx",
    "n_dx_codes", "n_ndc", "n_proc_codes", "n_active_months",
]
# Light recency/trend. Also removed in the ablation (contact-timing == access signal).
TEMPORAL_FEATURES = [
    "t_claims_recent3", "t_claims_prior9", "t_months_since_last",
    "t_active_recent3", "t_accel",
]
DEMO_FEATURES = ["age_at_index", "is_female", "is_male",
                 "reg_mw", "reg_ne", "reg_s", "reg_w"]


def _ident(prefix: str, raw: str, used: set) -> str:
    """Sanitize a code/class into a unique SQL-safe column name, prefixed by family."""
    s = re.sub(r"[^0-9a-z]+", "_", raw.lower()).strip("_")
    name = f"{prefix}_{s}"[:60]
    if name in used:                       # collision guard (rare; keep it loud-safe)
        i = 2
        while f"{name}_{i}" in used:
            i += 1
        name = f"{name}_{i}"
    used.add(name)
    return name


def _lit(v):
    return "'" + v.replace("'", "''") + "'"


def _views_sql(disease: str) -> str:
    cohort_pq = (DATA_PROCESSED / f"{disease}_cohort.parquet").as_posix()
    return f"""
    CREATE OR REPLACE MACRO m_abs(mid) AS ((CAST(mid AS BIGINT) // 100) * 12 + (CAST(mid AS BIGINT) % 100) - 1);
    CREATE OR REPLACE VIEW cohort AS SELECT * FROM read_parquet('{cohort_pq}');
    CREATE OR REPLACE VIEW claims AS SELECT * FROM {tbl('claims')};
    CREATE OR REPLACE VIEW rxlk AS
        SELECT upper(trim(NDC)) AS ndc, THPTC_CLAS_DESC, upper(USC_NAME) AS usc
        FROM {tbl('rx_lookup')};
    """


def _write_base(con, disease: str, tmp: Path):
    """Materialize base_claims / base_rx as IN-MEMORY tables (the original thin build
    proved these fit in RAM). Keeping them off disk leaves the tight local drive free for
    temp-spill and the per-family parquets."""
    excl = RX_TREATMENT_EXCLUDE_USC.get(disease, [])
    excl_sql = ", ".join(_lit(u) for u in excl) or "''"
    cls_excl = RX_TREATMENT_EXCLUDE_CLASS.get(disease, [])
    cls_sql = ", ".join(_lit(u) for u in cls_excl) or "''"
    # Prefer the pre-sliced baseline claims (Stage 3.0, resumable) — ~30M rows vs 1.3B —
    # so this stage is fast and disconnect-safe. Fall back to scanning raw claims.
    sliced = DATA_PROCESSED / f"{disease}_base_claims.parquet"
    if sliced.exists():
        con.execute(f"CREATE OR REPLACE TABLE base_claims AS "
                    f"SELECT * FROM read_parquet('{sliced.as_posix()}')")
    else:
        con.execute("""
            CREATE OR REPLACE TABLE base_claims AS
            SELECT c.PID AS PAT_ID, c.CODE_TYPE, upper(trim(c.CODE)) AS code_n,
                   m_abs(c.MONTH_ID) AS ev_abs, c.MONTH_ID,
                   m_abs(b.index_month) AS index_abs, b.split
            FROM claims c
            JOIN cohort b ON c.PID = b.PAT_ID
            WHERE m_abs(c.MONTH_ID) >= m_abs(b.FIRST_MONTH)
              AND m_abs(c.MONTH_ID) <  m_abs(b.index_month)""")
    con.execute(f"""
        CREATE OR REPLACE TABLE base_rx AS
        SELECT bc.PAT_ID, bc.split, r.THPTC_CLAS_DESC AS rxclass
        FROM base_claims bc
        JOIN rxlk r ON r.ndc = bc.code_n
        WHERE bc.CODE_TYPE = 'RX'
          AND r.THPTC_CLAS_DESC IS NOT NULL
          AND r.usc NOT IN ({excl_sql})
          AND r.THPTC_CLAS_DESC NOT IN ({cls_sql})""")


def _vocab(con, disease: str):
    regex = DISEASES[disease]["dx_regex"]
    dx = con.execute(f"""
        SELECT substr(code_n,1,3) AS cat, COUNT(DISTINCT PAT_ID) np
        FROM base_claims
        WHERE CODE_TYPE='DX' AND split='train'
          AND NOT regexp_matches(substr(code_n,1,3), '{regex}')
        GROUP BY 1 ORDER BY np DESC, cat LIMIT {RICH_DX_VOCAB_K}
    """).df()["cat"].tolist()
    proc = con.execute(f"""
        SELECT code_n AS code, COUNT(DISTINCT PAT_ID) np
        FROM base_claims WHERE CODE_TYPE='PROC' AND split='train'
        GROUP BY 1 ORDER BY np DESC, code LIMIT {RICH_PROC_VOCAB_K}
    """).df()["code"].tolist()
    rxc = con.execute("""
        SELECT rxclass, COUNT(DISTINCT PAT_ID) np
        FROM base_rx WHERE split='train'
        GROUP BY 1 ORDER BY np DESC, rxclass
    """).df()["rxclass"].tolist()
    return dx, proc, rxc


def _copy(con, tmp: Path, name: str, select: str):
    con.execute(f"COPY ({select}) TO '{(tmp / (name + '.parquet')).as_posix()}' (FORMAT PARQUET)")


def _write_families(con, dx_map, proc_map, rx_map, tmp: Path):
    """One streaming COPY per aggregate -> parquet. Each COUNT(DISTINCT) gets its OWN
    single-distinct pass: with only ~10 GB RAM, cramming four distincts (esp. distinct
    NDCs) into one pass spills >13 GB of temp. Splitting bounds peak temp; the passes
    re-scan the in-RAM base_claims, which is cheap."""
    cm_cols = [f"MAX(CASE WHEN CODE_TYPE='DX' AND regexp_matches(code_n, '{rx}') "
               f"THEN 1 ELSE 0 END) AS {name}" for name, rx in COMORBIDITIES.items()]

    # Plain sums + temporal + comorbidity (no DISTINCT) -> cheap
    _copy(con, tmp, "agg_sum", f"""
        SELECT PAT_ID,
               COUNT(*)                                          AS n_claims,
               SUM(CASE WHEN CODE_TYPE='DX'   THEN 1 ELSE 0 END) AS n_dx,
               SUM(CASE WHEN CODE_TYPE='PROC' THEN 1 ELSE 0 END) AS n_proc,
               SUM(CASE WHEN CODE_TYPE='RX'   THEN 1 ELSE 0 END) AS n_rx,
               SUM(CASE WHEN ev_abs >= index_abs-3 THEN 1 ELSE 0 END) AS t_claims_recent3,
               SUM(CASE WHEN ev_abs <  index_abs-3 THEN 1 ELSE 0 END) AS t_claims_prior9,
               MIN(index_abs) - 1 - MAX(ev_abs)                  AS t_months_since_last,
               {',' .join(cm_cols)}
        FROM base_claims GROUP BY PAT_ID""")

    # Month-based distincts (low cardinality, cheap)
    _copy(con, tmp, "agg_month", """
        SELECT PAT_ID,
               COUNT(DISTINCT MONTH_ID) AS n_active_months,
               COUNT(DISTINCT CASE WHEN ev_abs >= index_abs-3 THEN MONTH_ID END) AS t_active_recent3
        FROM base_claims GROUP BY PAT_ID""")

    # Code-based distincts: one distinct per pass (NDC is the expensive one)
    _copy(con, tmp, "agg_dxcodes",
          "SELECT PAT_ID, COUNT(DISTINCT code_n) AS n_dx_codes FROM base_claims WHERE CODE_TYPE='DX' GROUP BY PAT_ID")
    _copy(con, tmp, "agg_ndc",
          "SELECT PAT_ID, COUNT(DISTINCT code_n) AS n_ndc FROM base_claims WHERE CODE_TYPE='RX' GROUP BY PAT_ID")
    _copy(con, tmp, "agg_proccodes",
          "SELECT PAT_ID, COUNT(DISTINCT code_n) AS n_proc_codes FROM base_claims WHERE CODE_TYPE='PROC' GROUP BY PAT_ID")

    # Bag-of-codes as compact LONG tables (patient, code, count) -> pivoted to wide in
    # pandas. A 595k x 500 WIDE aggregate in DuckDB fills this machine's disk; the long
    # form is ~9M tiny rows and pivots to a ~1 GB matrix that fits in RAM.
    dx_in = ", ".join(_lit(c) for _, c in dx_map) or "''"
    proc_in = ", ".join(_lit(c) for _, c in proc_map) or "''"
    _copy(con, tmp, "dx_long", f"""
        SELECT PAT_ID, substr(code_n,1,3) AS code, COUNT(*) AS cnt
        FROM base_claims WHERE CODE_TYPE='DX' AND substr(code_n,1,3) IN ({dx_in})
        GROUP BY 1,2""")
    _copy(con, tmp, "proc_long", f"""
        SELECT PAT_ID, code_n AS code, COUNT(*) AS cnt
        FROM base_claims WHERE CODE_TYPE='PROC' AND code_n IN ({proc_in})
        GROUP BY 1,2""")
    _copy(con, tmp, "rx_long",
          "SELECT PAT_ID, rxclass AS code, COUNT(*) AS cnt FROM base_rx GROUP BY 1,2")


def _pivot(tmp: Path, long_name: str, code_to_ident: dict, index):
    """Pivot a compact long table (PAT_ID, code, cnt) to a wide int16 frame reindexed to
    the full cohort. Codes absent from the vocab are ignored; missing patients -> 0."""
    import pandas as pd
    lp = tmp / f"{long_name}.parquet"
    cols = list(code_to_ident.values())
    if not cols:
        return pd.DataFrame(index=index)
    df = pd.read_parquet(lp)
    df = df[df["code"].isin(code_to_ident)]
    wide = (df.pivot_table(index="PAT_ID", columns="code", values="cnt",
                           aggfunc="sum", fill_value=0)
              .rename(columns=code_to_ident))
    wide = wide.reindex(index=index, columns=cols, fill_value=0)
    return wide.astype("int16")


def _assemble(disease, tmp: Path, dx_map, proc_map, rx_map, out_pq: Path):
    """Build the wide feature matrix in pandas RAM (final matrix ~1 GB) from the narrow
    aggregate parquets + the pivoted bag long-tables. Column order matches meta['full']."""
    import numpy as np
    import pandas as pd

    coh = pd.read_parquet(DATA_PROCESSED / f"{disease}_cohort.parquet",
                          columns=["PAT_ID", "DER_SEX", "PAT_REGION", "age_at_index",
                                   "index_year", "split", "label"]).set_index("PAT_ID")
    idx = coh.index
    out = pd.DataFrame(index=idx)
    out["index_year"] = coh["index_year"]; out["split"] = coh["split"]; out["label"] = coh["label"]

    # demographics
    out["age_at_index"] = coh["age_at_index"]
    out["is_female"] = (coh["DER_SEX"] == "F").astype("int8")
    out["is_male"] = (coh["DER_SEX"] == "M").astype("int8")
    for r in ("MW", "NE", "S", "W"):
        out[f"reg_{r.lower()}"] = (coh["PAT_REGION"] == r).astype("int8")

    # narrow aggregates (merge, fill 0)
    def nar(name, cols):
        d = pd.read_parquet(tmp / f"{name}.parquet").set_index("PAT_ID")
        return d.reindex(index=idx)[cols]
    s = nar("agg_sum", ["n_claims", "n_dx", "n_proc", "n_rx",
                        "t_claims_recent3", "t_claims_prior9", "t_months_since_last"]
            + list(COMORBIDITIES))
    m = nar("agg_month", ["n_active_months", "t_active_recent3"])
    dc = nar("agg_dxcodes", ["n_dx_codes"]); nc = nar("agg_ndc", ["n_ndc"])
    pc = nar("agg_proccodes", ["n_proc_codes"])

    for c in ["n_claims", "n_dx", "n_proc", "n_rx"]:
        out[c] = s[c].fillna(0)
    out["n_dx_codes"] = dc["n_dx_codes"].fillna(0)
    out["n_ndc"] = nc["n_ndc"].fillna(0)
    out["n_proc_codes"] = pc["n_proc_codes"].fillna(0)
    out["n_active_months"] = m["n_active_months"].fillna(0)
    out["t_claims_recent3"] = s["t_claims_recent3"].fillna(0)
    out["t_claims_prior9"] = s["t_claims_prior9"].fillna(0)
    out["t_months_since_last"] = s["t_months_since_last"].fillna(12)
    out["t_active_recent3"] = m["t_active_recent3"].fillna(0)
    out["t_accel"] = out["t_claims_recent3"] - out["t_claims_prior9"] / 3.0
    for c in COMORBIDITIES:
        out[c] = s[c].fillna(0).astype("int8")

    # bags (pivot long -> wide), in meta order: dx, rx, proc
    dx_w = _pivot(tmp, "dx_long", {c: i for i, c in dx_map}, idx)
    rx_w = _pivot(tmp, "rx_long", {c: i for i, c in rx_map}, idx)
    px_w = _pivot(tmp, "proc_long", {c: i for i, c in proc_map}, idx)
    out = pd.concat([out, dx_w, rx_w, px_w], axis=1)

    # downcast the count/util block to keep the file compact
    util_int = (["n_claims", "n_dx", "n_proc", "n_rx", "n_dx_codes", "n_ndc",
                 "n_proc_codes", "n_active_months", "t_claims_recent3", "t_claims_prior9",
                 "t_months_since_last", "t_active_recent3"])
    for c in util_int:
        out[c] = out[c].astype("int32")
    out["t_accel"] = out["t_accel"].astype("float32")
    out["age_at_index"] = out["age_at_index"].astype("int16")
    out.reset_index().to_parquet(out_pq, index=False)


def build(disease: str = "arthritis"):
    if disease not in DISEASES:
        raise KeyError(f"unknown disease {disease!r}")
    tmp = Path(tempfile.mkdtemp(prefix=f"{disease}_feats_"))
    con = connect()
    # Machine reality: ~16.9 GB RAM (~10 free), 1 drive with ~25 GB free. Cap memory below
    # the OS-available RAM so DuckDB spills instead of thrashing, and raise the temp cap
    # explicitly (it otherwise auto-caps ~13 GB and the distinct passes overrun it).
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")
    con.execute(f"SET memory_limit='{_mem_limit_gb()}GB'")
    con.execute("SET max_temp_directory_size='22GiB'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    try:
        con.execute(_views_sql(disease))
        _write_base(con, disease, tmp)

        dx_vocab, proc_vocab, rx_vocab = _vocab(con, disease)
        used: set = set()
        dx_map = [(_ident("dx", c, used), c) for c in dx_vocab]
        proc_map = [(_ident("px", c, used), c) for c in proc_vocab]
        rx_map = [(_ident("rx", c, used), c) for c in rx_vocab]

        _write_families(con, dx_map, proc_map, rx_map, tmp)

        # ---- Leakage audits ---------------------------------------------------
        bad = con.execute(
            "SELECT COUNT(*) FROM base_claims WHERE ev_abs >= index_abs").fetchone()[0]
        assert bad == 0, f"LEAKAGE: {bad} baseline claims dated at/after index"
        regex = DISEASES[disease]["dx_regex"]
        pheno = [c for _, c in dx_map if con.execute(
            f"SELECT regexp_matches('{c}', '{regex}')").fetchone()[0]]
        assert not pheno, f"LEAKAGE: phenotype codes entered dx_bag vocab: {pheno}"

        dx_names = [i for i, _ in dx_map]
        proc_names = [i for i, _ in proc_map]
        rx_names = [i for i, _ in rx_map]
        out_pq = DATA_PROCESSED / f"{disease}_features.parquet"
        rx_drop = con.execute(                       # for the report, before dropping bases
            "SELECT (SELECT COUNT(*) FROM base_claims WHERE CODE_TYPE='RX')"
            " - (SELECT COUNT(*) FROM base_rx)").fetchone()[0]
        con.execute("DROP TABLE base_claims")        # free RAM before the pandas pivot
        con.execute("DROP TABLE base_rx")
        _assemble(disease, tmp, dx_map, proc_map, rx_map, out_pq)
        con.execute(f"CREATE OR REPLACE VIEW feats AS SELECT * FROM read_parquet('{out_pq.as_posix()}')")

        # ---- Family tags + derived column sets Stage 6 selects on -------------
        families = {
            "demo": DEMO_FEATURES,
            "comorbidity": list(COMORBIDITIES),
            "utilization": UTILIZATION_FEATURES,
            "temporal": TEMPORAL_FEATURES,
            "dx_bag": dx_names,
            "rx_class": rx_names,
            "proc_bag": proc_names,
        }
        removable = UTILIZATION_FEATURES + TEMPORAL_FEATURES       # access-confound ablation
        full = (DEMO_FEATURES + list(COMORBIDITIES) + UTILIZATION_FEATURES
                + TEMPORAL_FEATURES + dx_names + rx_names + proc_names)
        clinical = [c for c in full if c not in removable]         # no_util arm
        thin = DEMO_FEATURES + list(COMORBIDITIES) + UTILIZATION_FEATURES  # original 27
        meta = {"disease": disease, "features": full, "families": families,
                "removable_access": removable, "utilization": UTILIZATION_FEATURES,
                "clinical": clinical, "thin": thin, "n_features": len(full)}
        (DATA_PROCESSED / f"{disease}_features_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")

        vocab = {"disease": disease, "dx_k": len(dx_map), "proc_k": len(proc_map),
                 "rx_classes_k": len(rx_map),
                 "rx_excluded_usc": RX_TREATMENT_EXCLUDE_USC.get(disease, []),
                 "rx_excluded_class": RX_TREATMENT_EXCLUDE_CLASS.get(disease, []),
                 "dx_cats": {i: c for i, c in dx_map},
                 "proc_codes": {i: c for i, c in proc_map},
                 "rx_class_names": {i: c for i, c in rx_map}}
        (DATA_PROCESSED / f"{disease}_features_vocab.json").write_text(
            json.dumps(vocab, indent=2), encoding="utf-8")

        _report(con, disease, meta, dx_map, rx_map, rx_drop, out_pq)
        n = con.execute("SELECT COUNT(*) FROM feats").fetchone()[0]
        print(f"[features] {disease}: {n:,} rows, {len(full)} features "
              f"(demo {len(DEMO_FEATURES)}/cm {len(COMORBIDITIES)}/util {len(UTILIZATION_FEATURES)}/"
              f"temporal {len(TEMPORAL_FEATURES)}/dx {len(dx_names)}/rx {len(rx_names)}/"
              f"proc {len(proc_names)}) -> {out_pq}")
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _report(con, disease, meta, dx_map, rx_map, rx_drop, out_pq):
    n = con.execute("SELECT COUNT(*) FROM feats").fetchone()[0]
    zero = con.execute("SELECT COUNT(*) FROM feats WHERE n_claims=0").fetchone()[0]
    probe = [i for i, _ in dx_map[:5]] + [i for i, _ in rx_map[:5]]
    sig = con.execute(
        "SELECT label, " + ", ".join(f"ROUND(AVG({c}),3) {c}" for c in probe)
        + " FROM feats GROUP BY label ORDER BY label").df()
    fam = meta["families"]
    L = []; w = L.append
    w(f"# Stage 3.1 — Enriched features: {DISEASES[disease]['label']}")
    w("")
    w("_Fair-baseline Arm A inputs (see reports/fairness_contract.md). Bags chosen "
      "label-free from the train pool, frozen. Leakage + phenotype-exclusion audits passed._")
    w("")
    w(f"- **Rows:** {n:,}. **Features:** {meta['n_features']} across 7 families.")
    w(f"- **No-baseline-claim patients:** {zero:,} (kept, features=0).")
    w(f"- **RX claims dropped as disease-specific treatment:** {rx_drop:,} "
      f"(USC in {RX_TREATMENT_EXCLUDE_USC.get(disease, [])}).")
    w("")
    w("## Feature families")
    w("| family | count | in access-confound ablation |")
    w("|---|---|---|")
    keep = {"demo", "comorbidity", "dx_bag", "rx_class", "proc_bag"}
    for k, v in fam.items():
        w(f"| {k} | {len(v)} | {'kept' if k in keep else 'REMOVED'} |")
    w("")
    w("## Derived column sets (Stage 6 selects on these)")
    w(f"- **full** (rich): {len(meta['features'])}  ·  **clinical** (no-util): "
      f"{len(meta['clinical'])}  ·  **thin** (original 27): {len(meta['thin'])}")
    w("")
    w("## Signal check — mean count by outcome (top new features)")
    w("| feature | negatives | positives |")
    w("|---|---|---|")
    neg = sig[sig.label == 0].iloc[0]; pos = sig[sig.label == 1].iloc[0]
    for c in probe:
        w(f"| {c} | {neg[c]} | {pos[c]} |")
    w("")
    w(f"Table: `data/processed/{disease}_features.parquet` (gitignored). "
      f"Tags: `{disease}_features_meta.json`. Frozen vocab: `{disease}_features_vocab.json`.")
    (REPORTS / f"stage3_1_{disease}_features.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")
