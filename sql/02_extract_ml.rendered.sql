-- =============================================================================
-- 02_extract_ml.sql — THE SINGLE IQVIA EXTRACTION (ML + epidemiology)
-- IQVIA PharMetrics Plus (for Academics). Templated; render with:
--     python sql/render.py
-- and paste the rendered file (sql/02_extract_ml.rendered.sql) into Snowflake.
--
-- WHAT THIS DOES (one pull, raw + wide; every downstream choice stays local):
--   * Draws a reproducible hash sample of ELIGIBLE patients (>= 24
--     months continuous enrollment) — ~6% of the eligible population.
--   * Exports four Parquet tables for the sampled cohort:
--       patients   — one row/patient: birth year, sex, zip3/state, enroll summary
--       enrollment — continuous coverage spans (censoring + flexible washout)
--       claims     — de-duplicated (patient, date, code) stream: DX + PROC [+ RX]
--                    -> the longitudinal sequence for feature eng + Tier-3 models
--       events     — first/second qualifying arthritis dx per subtype (labels+epi)
--
-- >>> THE ONLY THING TO VERIFY: the identifiers below are the STANDARD
-- >>> PharMetrics layout. If the share spells any differently, just edit them
-- >>> inline (or in config/schema_config.yaml + re-render). Everything else is
-- >>> reversible locally, so only these 12 names must be right:
--   database/schema : PHARMETRICS.PUBLIC
--   tables          : ENROLLMENT, MEDICAL_CLAIMS, PHARMACY_CLAIMS, PATIENT
--   columns         : PAT_ID, MONTH_ID, FROM_DT, DIAG,
--                     PROC_CDE, NDC, DOBYR, DER_SEX, PAT_ZIP3
--
-- OPTIONAL 10-SECOND NAME CHECK (run first if unsure they match the share):
--   SELECT TABLE_NAME, ROW_COUNT FROM PHARMETRICS.INFORMATION_SCHEMA.TABLES
--     ORDER BY ROW_COUNT DESC;
--   SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM PHARMETRICS.INFORMATION_SCHEMA.COLUMNS
--     WHERE TABLE_NAME IN ('ENROLLMENT','MEDICAL_CLAIMS','PHARMACY_CLAIMS','PATIENT')
--     ORDER BY TABLE_NAME, ORDINAL_POSITION;
-- =============================================================================

-- 0) Reproducible sample of eligible patients --------------------------------
-- Eligibility (room for >=12mo baseline + a prediction window): keep patients
-- whose LONGEST continuous coverage span is >= 24 months.
-- Continuous = enrollment gaps <= 45 days (gaps-and-islands).
CREATE OR REPLACE TEMP TABLE _spans_all AS
WITH monthly AS (
  SELECT PAT_ID AS pid, MONTH_ID AS month
  FROM   PHARMETRICS.PUBLIC.ENROLLMENT
  WHERE  MONTH_ID BETWEEN '2017-01-01' AND '2024-12-31'
),
flagged AS (
  SELECT pid, month,
         CASE WHEN DATEDIFF('day',
                LAG(month) OVER (PARTITION BY pid ORDER BY month), month)
                > 45 THEN 1 ELSE 0 END AS new_span
  FROM monthly
),
grouped AS (
  SELECT pid, month,
         SUM(new_span) OVER (PARTITION BY pid ORDER BY month) AS span_id
  FROM flagged
)
SELECT pid, span_id, MIN(month) AS enroll_start, MAX(month) AS enroll_end,
       DATEDIFF('month', MIN(month), MAX(month)) AS span_months
FROM grouped
GROUP BY pid, span_id;

CREATE OR REPLACE TEMP TABLE cohort_ids AS
SELECT DISTINCT pid
FROM   _spans_all
WHERE  span_months >= 24
  -- deterministic ~6% sample, stable across re-runs:
  AND   MOD(ABS(HASH(pid)), 100) < 6;

-- 1) patients (denominator + demographics) -----------------------------------
CREATE OR REPLACE TEMP TABLE patients AS
SELECT  c.pid,
        d.DOBYR AS birth_year,
        d.DER_SEX        AS sex,
        d.PAT_ZIP3       AS zip3,          -- mapped to census region LOCALLY
        MIN(s.enroll_start) AS first_enroll,
        MAX(s.enroll_end)   AS last_enroll,
        MAX(s.span_months)  AS max_span_months,
        COUNT(*)            AS n_spans
FROM    cohort_ids c
JOIN    PHARMETRICS.PUBLIC.PATIENT d ON d.PAT_ID = c.pid
JOIN    _spans_all s ON s.pid = c.pid
GROUP BY c.pid, d.DOBYR, d.DER_SEX, d.PAT_ZIP3;

-- 2) enrollment spans (censoring + locally-tunable washout) -------------------
CREATE OR REPLACE TEMP TABLE enrollment AS
SELECT s.pid, s.span_id, s.enroll_start, s.enroll_end, s.span_months
FROM   _spans_all s JOIN cohort_ids c ON c.pid = s.pid;

-- 3) claims stream: de-duplicated (patient, date, code_type, code) ------------
-- Raw codes kept (truncate to 3-char locally if a smaller vocab is wanted).
-- RX block is optional: comment out the PHARMACY UNION if download size is a
-- concern — NDCs add predictive signal but the most rows.
CREATE OR REPLACE TEMP TABLE claims AS
SELECT DISTINCT pid, svc_date, code_type, code, pos FROM (
  -- diagnoses
  SELECT m.PAT_ID AS pid, m.FROM_DT AS svc_date,
         'DX'  AS code_type, m.DIAG AS code, m.POS AS pos
  FROM   PHARMETRICS.PUBLIC.MEDICAL_CLAIMS m
  JOIN   cohort_ids c ON c.pid = m.PAT_ID
  WHERE  m.FROM_DT BETWEEN '2017-01-01' AND '2024-12-31'
    AND  m.DIAG IS NOT NULL
  UNION ALL
  -- procedures
  SELECT m.PAT_ID, m.FROM_DT,
         'PROC', m.PROC_CDE, m.POS
  FROM   PHARMETRICS.PUBLIC.MEDICAL_CLAIMS m
  JOIN   cohort_ids c ON c.pid = m.PAT_ID
  WHERE  m.FROM_DT BETWEEN '2017-01-01' AND '2024-12-31'
    AND  m.PROC_CDE IS NOT NULL
  UNION ALL
  -- pharmacy (NDC) -- OPTIONAL: comment out this block to shrink the export
  SELECT p.PAT_ID, p.FROM_DT,
         'RX', p.NDC, NULL
  FROM   PHARMETRICS.PUBLIC.PHARMACY_CLAIMS p
  JOIN   cohort_ids c ON c.pid = p.PAT_ID
  WHERE  p.FROM_DT BETWEEN '2017-01-01' AND '2024-12-31'
    AND  p.NDC IS NOT NULL
);

-- 4) arthritis events (outcome labels + epi numerator) ------------------------
-- Full ICD-10 codes used here (L40.5 needs 4 chars). One row/patient/subtype.
CREATE OR REPLACE TEMP TABLE _dx AS
SELECT  m.PAT_ID AS pid, m.FROM_DT AS dx_date,
        CASE
          WHEN m.DIAG LIKE 'M15%' OR m.DIAG LIKE 'M16%'
            OR m.DIAG LIKE 'M17%' OR m.DIAG LIKE 'M18%'
            OR m.DIAG LIKE 'M19%' THEN 'OA'
          WHEN m.DIAG LIKE 'M05%' OR m.DIAG LIKE 'M06%' THEN 'RA'
          WHEN m.DIAG LIKE 'M10%' THEN 'GOUT'
          WHEN m.DIAG LIKE 'L40.5%' OR m.DIAG LIKE 'M07%' THEN 'PSA'
          WHEN m.DIAG LIKE 'M45%' THEN 'AS'
          WHEN m.DIAG LIKE 'M08%' THEN 'JUV'
          WHEN m.DIAG LIKE 'M12%' OR m.DIAG LIKE 'M13%' THEN 'OTHER'
        END AS subtype
FROM    PHARMETRICS.PUBLIC.MEDICAL_CLAIMS m
JOIN    cohort_ids c ON c.pid = m.PAT_ID
WHERE  (m.DIAG LIKE 'M0%' OR m.DIAG LIKE 'M1%'
     OR m.DIAG LIKE 'M45%' OR m.DIAG LIKE 'L40.5%');

CREATE OR REPLACE TEMP TABLE events AS
WITH distinct_days AS (
  SELECT DISTINCT pid, subtype, dx_date FROM _dx WHERE subtype IS NOT NULL
),
firsts AS (
  SELECT pid, subtype, MIN(dx_date) AS first_dx, COUNT(*) AS n_claim_days
  FROM distinct_days GROUP BY pid, subtype
)
SELECT f.pid, f.subtype, f.first_dx, f.n_claim_days,
       (SELECT MIN(dd.dx_date) FROM distinct_days dd
         WHERE dd.pid = f.pid AND dd.subtype = f.subtype
           AND dd.dx_date >= DATEADD('day', 30, f.first_dx)
       ) AS second_qualifying_dx
FROM firsts f;

-- 5) QC — eyeball BEFORE exporting (paste this back if anything looks off) ----
SELECT 'patients'      AS tbl, COUNT(*)::STRING AS n FROM patients
UNION ALL SELECT 'enrollment_spans', COUNT(*)::STRING FROM enrollment
UNION ALL SELECT 'claims_rows',      COUNT(*)::STRING FROM claims
UNION ALL SELECT 'event_rows',       COUNT(*)::STRING FROM events
UNION ALL SELECT 'birth_year_min',   MIN(birth_year)::STRING FROM patients
UNION ALL SELECT 'birth_year_max',   MAX(birth_year)::STRING FROM patients;
SELECT sex, COUNT(*) AS n FROM patients GROUP BY sex ORDER BY sex;
SELECT LEFT(zip3,1) AS zip_region_digit, COUNT(*) AS n
  FROM patients GROUP BY 1 ORDER BY 1;                 -- geographic spread sane?
SELECT code_type, COUNT(*) AS n FROM claims GROUP BY code_type;   -- DX/PROC/RX present?
SELECT subtype, COUNT(*) AS n FROM events GROUP BY subtype ORDER BY subtype;

-- 6) EXPORT to Parquet (download to data/raw/) --------------------------------
COPY INTO @~/iqvia/patients   FROM patients   FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE;
COPY INTO @~/iqvia/enrollment FROM enrollment FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE;
COPY INTO @~/iqvia/claims     FROM claims     FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE MAX_FILE_SIZE=268435456;
COPY INTO @~/iqvia/events     FROM events     FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE;
-- GET @~/iqvia/patients   file:///<path>/data/raw/;
-- GET @~/iqvia/enrollment file:///<path>/data/raw/;
-- GET @~/iqvia/claims     file:///<path>/data/raw/;
-- GET @~/iqvia/events     file:///<path>/data/raw/;
-- (Or use the Snowflake UI "Download" on each table. claims is the largest.)
