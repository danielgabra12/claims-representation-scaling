"""Out-of-core access to the raw Parquet sample via DuckDB.

Discipline (non-negotiable, from the handoff): the big files are QUERIED, never
loaded whole. `claims_2m.parquet` is 290M rows / 2.3 GB; a `pd.read_parquet` of it
will thrash the machine. Every helper here pushes work into DuckDB and returns
either a small aggregate DataFrame or a DuckDB relation the caller streams.

Usage:
    from src.scale.data.access import connect, tbl, q
    con = connect()
    df = q(con, f"SELECT CODE_TYPE, COUNT(*) FROM {tbl('claims')} GROUP BY 1")
"""
from __future__ import annotations

import duckdb

from src.scale.config import RAW


def connect(threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """A fresh in-memory DuckDB connection configured for quiet, out-of-core work.

    In-memory is correct here: DuckDB reads the Parquet directly from disk and only
    materializes the (small) query result, so there is no benefit to a persistent
    .db file for read-only aggregation. Progress bars are disabled because they
    flood non-interactive tool output.
    """
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA disable_progress_bar")
    if threads is not None:
        con.execute(f"PRAGMA threads={int(threads)}")
    return con


def tbl(name: str) -> str:
    """Return a parquet-scan expression for a raw table, e.g. tbl('claims').

    Yields a `read_parquet('...')` string safe to interpolate into SQL. Raises on
    unknown names so a typo fails loudly instead of scanning nothing.
    """
    if name not in RAW:
        raise KeyError(f"unknown raw table {name!r}; known: {sorted(RAW)}")
    path = RAW[name].as_posix()
    return f"read_parquet('{path}')"


def q(con: duckdb.DuckDBPyConnection, sql: str):
    """Run SQL and return a pandas DataFrame. Use ONLY for results that are small
    (aggregates, schemas, samples). For large intermediates, keep the DuckDB
    relation and write straight to Parquet with COPY instead of round-tripping
    through pandas.
    """
    return con.execute(sql).df()


def raw_exists() -> dict[str, bool]:
    """Which raw Parquet files are present locally (Stage 1 arrival check).

    RAW paths may be globs (the scale pull lands as many parts), so match by glob.
    """
    import glob as _glob
    return {name: bool(_glob.glob(str(path))) for name, path in RAW.items()}
