# Stage 0 — Data manifest (local Parquet sample)

_Generated 2026-07-05 by `src.scale.data.manifest`. All facts are DuckDB aggregates over the raw files; nothing is loaded whole._

This is a `MOD(HASH(PAT_ID),10)=0` ~2.2M-patient sample of the 22.4M population. The full population is aggregate-query-only in Snowflake (human-only extraction). Reaching the 3M+ ladder points requires that extraction; the local sample supports the pipeline and the lower ladder.

## Table shapes and schemas
- **patients** — 2,243,032 rows. Columns: `PAT_ID`, `FIRST_MONTH`, `LAST_MONTH`, `N_ENROLLED_MONTHS`, `DER_YOB`, `DER_SEX`, `PAT_STATE`, `PAT_ZIP3`, `PAT_REGION`, `DOD`
- **enrollment** — 60,133,074 rows. Columns: `PAT_ID`, `MONTH_ID`, `PAY_TYPE`
- **claims** — 289,834,118 rows. Columns: `PID`, `MONTH_ID`, `CODE_TYPE`, `CODE`
- **dx_lookup** — 106,773 rows. Columns: `DX_CD`, `DIAGNOSIS`, `DIAGNOSIS_DESC`, `DIAG_VERS_TYP_ID`
- **rx_lookup** — 261,363 rows. Columns: `NDC`, `PRODUCT_NAME`, `GENERIC_NAME`, `GPI14`, `GPI_DESC`, `GPI2`, `GPI2_DESC`, `THPTC_CLAS_ID`, `THPTC_CLAS_DESC`, `DOSAGE_FORM_NM`, `ROUTE`, `STRENGTH`, `USC_CD`, `USC_NAME`
- **pr_lookup** — 78,108 rows. Columns: `PROCEDURE_CD`, `PROCEDURE`, `PROCEDURE_DESC`, `PROCEDURE_TYPE_CD`, `PRC_VERS_TYP_ID`

## Claims content (the event substrate)
| CODE_TYPE | rows | distinct codes |
|---|---|---|
| PROC | 119,620,324 | 19,944 |
| DX | 117,844,698 | 51,644 |
| RX | 52,369,096 | 76,280 |

- MONTH_ID range: **201901–202504** (YYYYMM).
- Distinct patients with any claim: **1,894,230**.

## Key-join integrity
- Patients: **2,243,032**. Claims PID matching a PAT_ID: **1,894,230** of **1,894,230** distinct claim PIDs (1:1, no orphans).
- Patients with no claim (enrolled-only): **348,802**.

## Enrollment
- MONTH_ID range: **201901–202502**. Distinct enrolled patients: **2,243,032**.

## Demographics and data-quality flags
- DER_SEX: F=1,159,553, M=1,045,907, U=37,572
- PAT_REGION: S=701,979, MW=640,461, W=480,893, NE=365,392, (null)=54,307
- DER_YOB range: **0–2025**; implausible (YOB<1900, i.e. missing/0): **54,290** — excluded at cohort build.

**Guards enforced at Stage 2:** drop YOB<1900; `DER_SEX=U` and null region kept as explicit categories (not imputed).
