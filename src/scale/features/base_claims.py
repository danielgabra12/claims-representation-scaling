"""Stage 3.0 — resumable baseline-window claims slice (feeds features + sequences).

Run: python -m src.scale.features.base_claims arthritis

The raw claims are ~1.3B rows across 64 Parquet parts. Feature/sequence building only
needs each cohort patient's claims in their BASELINE window [FIRST_MONTH, index). This
stage scans the 64 parts ONE AT A TIME, writing a small filtered part each time, and
SKIPS parts already done — so a Colab disconnect costs only the current part, and
re-running resumes. The ~30M-row result (`<disease>_base_claims.parquet`, ~40x smaller
than raw) is what features/sequences read, making those stages fast and disconnect-safe.
"""
import glob
import os
import shutil
import sys
from pathlib import Path

from src.scale.config import DISEASES, DATA_PROCESSED, RAW, mem_limit_gb
from src.scale.data.access import connect

_DRIVE = os.environ.get("SURF_DRIVE_OUT")   # checkpoint each finished part here (Colab)


def _drive_parts_dir(disease):
    return Path(_DRIVE) / "processed" / f"{disease}_base_claims_parts" if _DRIVE else None


def build(disease: str = "arthritis"):
    if disease not in DISEASES:
        raise KeyError(f"unknown disease {disease!r}")
    cohort_pq = (DATA_PROCESSED / f"{disease}_cohort.parquet").as_posix()
    parts = sorted(glob.glob(str(RAW["claims"])))
    if not parts:
        raise FileNotFoundError(f"no claims parts match {RAW['claims']}")
    partsdir = DATA_PROCESSED / f"{disease}_base_claims_parts"
    partsdir.mkdir(parents=True, exist_ok=True)
    # restore any parts finished in a prior (disconnected) session
    ddir = _drive_parts_dir(disease)
    if ddir and ddir.is_dir():
        for f in ddir.glob("part_*.parquet"):
            if not (partsdir / f.name).exists():
                shutil.copy(f, partsdir / f.name)

    con = connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")
    con.execute(f"SET memory_limit='{mem_limit_gb()}GB'")
    con.execute("CREATE MACRO m_abs(mid) AS "
                "((CAST(mid AS BIGINT)//100)*12 + (CAST(mid AS BIGINT)%100) - 1)")
    con.execute(f"""CREATE VIEW cohort AS
        SELECT PAT_ID, m_abs(FIRST_MONTH) AS first_abs,
               m_abs(index_month) AS index_abs, split
        FROM read_parquet('{cohort_pq}')""")

    for i, part in enumerate(parts):
        outp = partsdir / f"part_{i:03d}.parquet"
        if outp.exists() and outp.stat().st_size > 0:          # resume: skip done parts
            continue
        con.execute(f"""COPY (
            SELECT c.PID AS PAT_ID, c.CODE_TYPE, upper(trim(c.CODE)) AS code_n,
                   m_abs(c.MONTH_ID) AS ev_abs, c.MONTH_ID, b.index_abs, b.split
            FROM read_parquet('{Path(part).as_posix()}') c
            JOIN cohort b ON c.PID = b.PAT_ID
            WHERE m_abs(c.MONTH_ID) >= b.first_abs
              AND m_abs(c.MONTH_ID) <  b.index_abs
        ) TO '{outp.as_posix()}' (FORMAT PARQUET)""")
        if ddir:                                    # persist this part to Drive now
            ddir.mkdir(parents=True, exist_ok=True)
            shutil.copy(outp, ddir / outp.name)
        print(f"  base_claims {disease}: part {i+1}/{len(parts)}", flush=True)

    out = DATA_PROCESSED / f"{disease}_base_claims.parquet"
    con.execute(f"""COPY (SELECT * FROM read_parquet('{(partsdir/'part_*.parquet').as_posix()}'))
        TO '{out.as_posix()}' (FORMAT PARQUET)""")
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    print(f"[base_claims] {disease}: {n:,} baseline-window rows -> {out}", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "arthritis")
