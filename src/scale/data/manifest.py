"""Data manifest / arrival check.

Run: python -m src.scale.data.manifest

Recomputes the structural facts the pipeline depends on (row counts, schemas,
date ranges, code coverage, key-join integrity, data-quality flags) and writes
reports/stage0_data_manifest.md. Intentionally reads nothing whole -- every fact
is a DuckDB aggregate. Doubles as the spec Stage 1 "verify on arrival" gate: point
it at a new data/raw/ and it re-validates before any modeling.
"""
from __future__ import annotations

import datetime as dt

from src.scale.config import REPORTS
from src.scale.data.access import connect, tbl, raw_exists

OUT = REPORTS / "stage0_data_manifest.md"


def _one(con, sql):
    return con.execute(sql).fetchone()


def build() -> str:
    present = raw_exists()
    missing = [k for k, v in present.items() if not v]
    if missing:
        raise FileNotFoundError(f"missing raw parquet: {missing}")

    con = connect()
    lines: list[str] = []
    w = lines.append

    w("# Stage 0 — Data manifest (local Parquet sample)")
    w("")
    w(f"_Generated {dt.date.today().isoformat()} by `src.scale.data.manifest`. "
      "All facts are DuckDB aggregates over the raw files; nothing is loaded whole._")
    w("")
    w("This is a `MOD(HASH(PAT_ID),10)=0` ~2.2M-patient sample of the 22.4M "
      "population. The full population is aggregate-query-only in Snowflake "
      "(human-only extraction). Reaching the 3M+ ladder points requires that "
      "extraction; the local sample supports the pipeline and the lower ladder.")
    w("")

    # --- Per-table shape + schema --------------------------------------------
    w("## Table shapes and schemas")
    for name in ("patients", "enrollment", "claims", "dx_lookup", "rx_lookup", "pr_lookup"):
        n = _one(con, f"SELECT COUNT(*) FROM {tbl(name)}")[0]
        cols = con.execute(f"DESCRIBE SELECT * FROM {tbl(name)}").df()
        colstr = ", ".join(f"`{c}`" for c in cols["column_name"])
        w(f"- **{name}** — {n:,} rows. Columns: {colstr}")
    w("")

    # --- Claims content ------------------------------------------------------
    w("## Claims content (the event substrate)")
    ct = con.execute(
        f"SELECT CODE_TYPE, COUNT(*) n, COUNT(DISTINCT CODE) ncodes "
        f"FROM {tbl('claims')} GROUP BY 1 ORDER BY 2 DESC").df()
    w("| CODE_TYPE | rows | distinct codes |")
    w("|---|---|---|")
    for r in ct.itertuples(index=False):
        w(f"| {r.CODE_TYPE} | {r.n:,} | {r.ncodes:,} |")
    mmin, mmax = _one(con, f"SELECT MIN(MONTH_ID), MAX(MONTH_ID) FROM {tbl('claims')}")
    ndp = _one(con, f"SELECT COUNT(DISTINCT PID) FROM {tbl('claims')}")[0]
    w("")
    w(f"- MONTH_ID range: **{mmin}–{mmax}** (YYYYMM).")
    w(f"- Distinct patients with any claim: **{ndp:,}**.")
    w("")

    # --- Key integrity -------------------------------------------------------
    w("## Key-join integrity")
    npat = _one(con, f"SELECT COUNT(*) FROM {tbl('patients')}")[0]
    matched = _one(
        con,
        f"SELECT COUNT(*) FROM (SELECT DISTINCT PID FROM {tbl('claims')}) c "
        f"JOIN {tbl('patients')} p ON c.PID = p.PAT_ID")[0]
    w(f"- Patients: **{npat:,}**. Claims PID matching a PAT_ID: **{matched:,}** "
      f"of **{ndp:,}** distinct claim PIDs "
      f"({'1:1, no orphans' if matched == ndp else 'ORPHANS PRESENT — investigate'}).")
    w(f"- Patients with no claim (enrolled-only): **{npat - ndp:,}**.")
    w("")

    # --- Enrollment ----------------------------------------------------------
    emin, emax = _one(con, f"SELECT MIN(MONTH_ID), MAX(MONTH_ID) FROM {tbl('enrollment')}")
    epat = _one(con, f"SELECT COUNT(DISTINCT PAT_ID) FROM {tbl('enrollment')}")[0]
    w("## Enrollment")
    w(f"- MONTH_ID range: **{emin}–{emax}**. Distinct enrolled patients: **{epat:,}**.")
    w("")

    # --- Demographics + data-quality flags -----------------------------------
    w("## Demographics and data-quality flags")
    sex = con.execute(
        f"SELECT DER_SEX, COUNT(*) n FROM {tbl('patients')} GROUP BY 1 ORDER BY 2 DESC").df()
    reg = con.execute(
        f"SELECT COALESCE(PAT_REGION,'(null)') PAT_REGION, COUNT(*) n "
        f"FROM {tbl('patients')} GROUP BY 1 ORDER BY 2 DESC").df()
    w("- DER_SEX: " + ", ".join(f"{r.DER_SEX}={r.n:,}" for r in sex.itertuples(index=False)))
    w("- PAT_REGION: " + ", ".join(f"{r.PAT_REGION}={r.n:,}" for r in reg.itertuples(index=False)))
    ymin, ymax = _one(con, f"SELECT MIN(DER_YOB), MAX(DER_YOB) FROM {tbl('patients')}")
    ybad = _one(con, f"SELECT COUNT(*) FROM {tbl('patients')} WHERE DER_YOB < 1900")[0]
    w(f"- DER_YOB range: **{ymin}–{ymax}**; implausible (YOB<1900, i.e. missing/0): "
      f"**{ybad:,}** — excluded at cohort build.")
    w("")
    w("**Guards enforced at Stage 2:** drop YOB<1900; `DER_SEX=U` and null region "
      "kept as explicit categories (not imputed).")
    w("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


if __name__ == "__main__":
    text = build()
    print(text)
    print(f"\n[written] {OUT}")
