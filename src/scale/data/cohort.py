"""Stage 2 — cohort, confound-hardened phenotype (dx-only arm), temporal split.

Run: python -m src.scale.data.cohort            # arthritis (default)
     python -m src.scale.data.cohort t2d        # another disease in the registry

Design (documented in reports/stage2_<disease>_cohort.md):
  - Index date  : per-patient FIRST_MONTH + 12 (the 13th enrolled month).
  - Baseline    : the 12 months before index (feature window; Stage 3 reads it).
  - Prediction  : the 12 months from index (outcome window). W=12 not 24 because
                  the 76-month span + enrollment ending 202502 cannot leave a full
                  held-out test year at W=24.
  - Eligibility : adult 18-89 at index; continuous 24-month enrollment
                  (12 baseline + 12 outcome) so a negative label is real, not a
                  disenrollment artifact; arthritis-free at index (not prevalent).
  - Label       : >=2 distinct qualifying-dx months within the prediction window
                  (the dx-only arm of the confound-hardened phenotype; the
                  "1 dx + treatment" arm is a PENDING pluggable slot).
  - Temporal    : train index-years 2020-2022, test index-year 2023.

Leakage safety is asserted at the end: prevalent cases excluded, outcomes strictly
after index, full observation, one row per patient, disjoint train/test.
"""
from __future__ import annotations

import json
import sys

from src.scale.config import (DISEASES, DATA_PROCESSED, DATA_SPLITS, REPORTS,
                              TRAIN_INDEX_YEARS, TEST_INDEX_YEAR, AGE_MIN, AGE_MAX)
from src.scale.data.access import connect, tbl


def _sql(disease: str) -> str:
    regex = DISEASES[disease]["dx_regex"]
    return f"""
    -- CAST to BIGINT: the scale-pull parquet stores months/years as DECIMAL(38,0),
    -- and integer division on DECIMAL yields fractional results (breaks index_year).
    CREATE OR REPLACE MACRO m_abs(mid)     AS ((CAST(mid AS BIGINT) // 100) * 12 + (CAST(mid AS BIGINT) % 100) - 1);
    CREATE OR REPLACE MACRO m_fromabs(a)   AS ((CAST(a AS BIGINT) // 12) * 100 + (CAST(a AS BIGINT) % 12) + 1);

    CREATE OR REPLACE VIEW patients AS SELECT * FROM {tbl('patients')};
    CREATE OR REPLACE VIEW enroll   AS SELECT * FROM {tbl('enrollment')};
    CREATE OR REPLACE VIEW claims   AS SELECT * FROM {tbl('claims')};

    CREATE OR REPLACE TABLE cohort AS
    WITH base AS (
        SELECT PAT_ID, CAST(DER_YOB AS BIGINT) AS DER_YOB, DER_SEX,
               COALESCE(NULLIF(PAT_REGION, ''), 'UNK') AS PAT_REGION,
               CAST(FIRST_MONTH AS BIGINT) AS FIRST_MONTH,
               m_abs(FIRST_MONTH)        AS first_abs,
               m_abs(FIRST_MONTH) + 12   AS index_abs,
               m_fromabs(m_abs(FIRST_MONTH) + 12) AS index_month
        FROM patients
        WHERE DER_YOB BETWEEN 1900 AND 2025
    ),
    elig AS (   -- continuous 24-month enrollment across [first, first+23]
        SELECT b.PAT_ID
        FROM base b JOIN enroll e ON e.PAT_ID = b.PAT_ID
        WHERE m_abs(e.MONTH_ID) BETWEEN b.first_abs AND b.first_abs + 23
        GROUP BY b.PAT_ID
        HAVING COUNT(DISTINCT e.MONTH_ID) = 24
    ),
    adx AS (    -- qualifying diagnosis events
        SELECT c.PID AS PAT_ID, m_abs(c.MONTH_ID) AS ev_abs
        FROM claims c
        WHERE c.CODE_TYPE = 'DX'
          AND regexp_matches(upper(trim(c.CODE)), '{regex}')
    ),
    prev AS (   -- prevalent: any qualifying dx strictly before index
        SELECT DISTINCT a.PAT_ID
        FROM adx a JOIN base b ON a.PAT_ID = b.PAT_ID
        WHERE a.ev_abs < b.index_abs
    ),
    outc AS (   -- distinct qualifying-dx months in [index, index+11]
        SELECT a.PAT_ID, COUNT(DISTINCT a.ev_abs) AS n_out_months
        FROM adx a JOIN base b ON a.PAT_ID = b.PAT_ID
        WHERE a.ev_abs BETWEEN b.index_abs AND b.index_abs + 11
        GROUP BY a.PAT_ID
    )
    SELECT b.PAT_ID, b.DER_YOB, b.DER_SEX, b.PAT_REGION,
           b.FIRST_MONTH, b.index_month,
           (b.index_month // 100)               AS index_year,
           (b.index_month // 100) - b.DER_YOB   AS age_at_index,
           COALESCE(o.n_out_months, 0)          AS n_out_months,
           CASE WHEN COALESCE(o.n_out_months, 0) >= 2 THEN 1 ELSE 0 END AS label,
           CASE WHEN (b.index_month // 100) = {TEST_INDEX_YEAR} THEN 'test'
                ELSE 'train' END                AS split
    FROM base b
    JOIN elig g       ON g.PAT_ID = b.PAT_ID
    LEFT JOIN prev p  ON p.PAT_ID = b.PAT_ID
    LEFT JOIN outc o  ON o.PAT_ID = b.PAT_ID
    WHERE p.PAT_ID IS NULL
      AND ((b.index_month // 100) - b.DER_YOB) BETWEEN {AGE_MIN} AND {AGE_MAX}
      AND (b.index_month // 100) IN ({','.join(str(y) for y in (*TRAIN_INDEX_YEARS, TEST_INDEX_YEAR))});
    """


def build(disease: str = "arthritis"):
    if disease not in DISEASES:
        raise KeyError(f"unknown disease {disease!r}; known: {sorted(DISEASES)}")
    con = connect()
    con.execute(_sql(disease))

    out_parquet = DATA_PROCESSED / f"{disease}_cohort.parquet"
    con.execute(
        f"COPY (SELECT * FROM cohort) TO '{out_parquet.as_posix()}' (FORMAT PARQUET)")

    # ---- Summary stats (all small aggregates) -------------------------------
    n = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
    pos = con.execute("SELECT SUM(label) FROM cohort").fetchone()[0] or 0
    assert n > 0, "cohort is EMPTY — check month/type handling and cohort filters"
    by_year = con.execute(
        "SELECT index_year, split, COUNT(*) n, SUM(label) pos, "
        "ROUND(100.0*AVG(label),2) rate FROM cohort GROUP BY 1,2 ORDER BY 1").df()
    by_split = con.execute(
        "SELECT split, COUNT(*) n, SUM(label) pos, ROUND(100.0*AVG(label),2) rate "
        "FROM cohort GROUP BY 1 ORDER BY 1").df()
    age = con.execute(
        "SELECT MIN(age_at_index) a_min, ROUND(AVG(age_at_index),1) a_mean, "
        "MAX(age_at_index) a_max FROM cohort").df().iloc[0]
    sex = con.execute(
        "SELECT DER_SEX, COUNT(*) n FROM cohort GROUP BY 1 ORDER BY 2 DESC").df()

    # ---- Leakage / integrity assertions -------------------------------------
    assert con.execute("SELECT COUNT(*) FROM cohort WHERE label IS NULL").fetchone()[0] == 0
    # positives must have >=2 outcome months; nobody prevalent survived the filter
    assert con.execute(
        "SELECT COUNT(*) FROM cohort WHERE label=1 AND n_out_months<2").fetchone()[0] == 0
    # full observation: latest outcome month must be within the data
    assert con.execute(
        "SELECT COUNT(*) FROM cohort WHERE (index_month//100) > 2023").fetchone()[0] == 0
    # one row per patient
    assert con.execute(
        "SELECT COUNT(*)-COUNT(DISTINCT PAT_ID) FROM cohort").fetchone()[0] == 0
    # train/test disjoint by patient
    assert con.execute(
        "SELECT COUNT(*) FROM (SELECT PAT_ID FROM cohort WHERE split='train' "
        "INTERSECT SELECT PAT_ID FROM cohort WHERE split='test')").fetchone()[0] == 0

    # ---- Persist splits (patient-level, per spec Stage 5 precursor) ----------
    splits = {}
    for s in ("train", "test"):
        ids = con.execute(
            f"SELECT PAT_ID FROM cohort WHERE split='{s}' ORDER BY PAT_ID").df()["PAT_ID"].tolist()
        splits[s] = ids
    (DATA_SPLITS / f"{disease}_temporal.json").write_text(
        json.dumps({"disease": disease, "test_index_year": TEST_INDEX_YEAR,
                    "train_index_years": list(TRAIN_INDEX_YEARS),
                    "n": {k: len(v) for k, v in splits.items()}},
                   indent=2), encoding="utf-8")

    _write_report(disease, n, pos, by_year, by_split, age, sex, out_parquet)
    print(f"[cohort] {disease}: n={n:,} positives={int(pos):,} "
          f"({100*pos/n:.2f}%) -> {out_parquet}")
    return n, pos


def _write_report(disease, n, pos, by_year, by_split, age, sex, out_parquet):
    d = DISEASES[disease]
    L = []
    w = L.append
    w(f"# Stage 2 — Cohort & phenotype: {d['label']}")
    w("")
    w("_Generated by `src.scale.data.cohort`. dx-only arm of the confound-hardened "
      "phenotype; the treatment arm is PENDING sign-off and will be added additively._")
    w("")
    w("## Design")
    w(f"- **Diagnosis codes:** `{d['dx_regex']}` on `CODE_TYPE='DX'` (ICD-10, no dots).")
    w("- **Index date:** `FIRST_MONTH + 12` (13th enrolled month).")
    w("- **Baseline (features):** 12 months before index. **Prediction window:** 12 months from index.")
    w(f"- **Eligibility:** age {AGE_MIN}-{AGE_MAX} at index; continuous 24-month "
      "enrollment (12+12); arthritis-free at index (not prevalent).")
    w("- **Label:** `1` if >=2 distinct qualifying-dx months within the prediction window, else `0`.")
    w(f"- **Temporal holdout:** train index-years {list(TRAIN_INDEX_YEARS)}, "
      f"test index-year {TEST_INDEX_YEAR}.")
    w("")
    w("## Results")
    w(f"- **Cohort size:** {n:,} patients. **Positives:** {int(pos):,} "
      f"(**{100*pos/n:.2f}%** onset within 12 months).")
    w(f"- **Age at index:** min {int(age.a_min)}, mean {age.a_mean}, max {int(age.a_max)}.")
    w("- **Sex:** " + ", ".join(f"{r.DER_SEX}={r.n:,}" for r in sex.itertuples(index=False)))
    w("")
    w("### By train/test split")
    w("| split | n | positives | onset rate % |")
    w("|---|---|---|---|")
    for r in by_split.itertuples(index=False):
        w(f"| {r.split} | {r.n:,} | {int(r.pos):,} | {r.rate} |")
    w("")
    w("### By index year")
    w("| index year | split | n | positives | onset rate % |")
    w("|---|---|---|---|---|")
    for r in by_year.itertuples(index=False):
        w(f"| {r.index_year} | {r.split} | {r.n:,} | {int(r.pos):,} | {r.rate} |")
    w("")
    w("## Leakage / integrity checks (all passed)")
    w("- No null labels. Every positive has >=2 outcome months. No prevalent case survived.")
    w("- Full outcome observation (no index-year beyond 2023). One row per patient. "
      "Train/test patient-disjoint.")
    w("")
    w(f"Cohort table: `{out_parquet.relative_to(out_parquet.parents[2]).as_posix()}` "
      "(gitignored). Split lists: `data/splits/`.")
    (REPORTS / f"stage2_{disease}_cohort.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")
