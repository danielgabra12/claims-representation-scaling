-- =============================================================================
-- 02_extract_ml.sql — THE SINGLE IQVIA EXTRACTION (ML + epidemiology)
-- IQVIA PharMetrics Plus (for Academics). Templated; render with:
--     python sql/render.py
-- and paste the rendered file (sql/02_extract_ml.rendered.sql) into Snowflake.
--
-- WHAT THIS DOES (one pull, raw + wide; every downstream choice stays local):
--   * Draws a reproducible hash sample of ELIGIBLE patients (>= ${min_total_months}
--     months continuous enrollment) — ~${sample_pct}% of the eligible population.
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
--   database/schema : ${database}.${schema}
--   tables          : ${enrollment}, ${claims}, ${pharmacy}, ${demographics}
--   columns         : ${patient_id}, ${elig_month}, ${claim_date}, ${icd_code},
--                     ${procedure_code}, ${ndc_code}, ${birth_year}, ${sex}, ${zip3}
--
-- OPTIONAL 10-SECOND NAME CHECK (run first if unsure they match the share):
--   SELECT TABLE_NAME, ROW_COUNT FROM ${database}.INFORMATION_SCHEMA.TABLES
--     ORDER BY ROW_COUNT DESC;
--   SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM ${database}.INFORMATION_SCHEMA.COLUMNS
--     WHERE TABLE_NAME IN ('${enrollment}','${claims}','${pharmacy}','${demographics}')
--     ORDER BY TABLE_NAME, ORDINAL_POSITION;
-- =============================================================================

-- 0) Reproducible sample of eligible patients --------------------------------
-- Eligibility (room for >=12mo baseline + a prediction window): keep patients
-- whose LONGEST continuous coverage span is >= ${min_total_months} months.
-- Continuous = enrollment gaps <= ${enrollment_gap_days} days (gaps-and-islands).
CREATE OR REPLACE TEMP TABLE _spans_all AS
WITH monthly AS (
  SELECT ${patient_id} AS pid, ${elig_month} AS month
  FROM   ${database}.${schema}.${enrollment}
  WHERE  ${elig_month} BETWEEN '${lookback_start}' AND '${end_date}'
),
flagged AS (
  SELECT pid, month,
         CASE WHEN DATEDIFF('day',
                LAG(month) OVER (PARTITION BY pid ORDER BY month), month)
                > ${enrollment_gap_days} THEN 1 ELSE 0 END AS new_span
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
WHERE  span_months >= ${min_total_months}
  -- deterministic ~${sample_pct}% sample, stable across re-runs:
  AND   MOD(ABS(HASH(pid)), 100) < ${sample_pct};

-- 1) patients (denominator + demographics) -----------------------------------
CREATE OR REPLACE TEMP TABLE patients AS
SELECT  c.pid,
        d.${birth_year} AS birth_year,
        d.${sex}        AS sex,
        d.${zip3}       AS zip3,          -- mapped to census region LOCALLY
        MIN(s.enroll_start) AS first_enroll,
        MAX(s.enroll_end)   AS last_enroll,
        MAX(s.span_months)  AS max_span_months,
        COUNT(*)            AS n_spans
FROM    cohort_ids c
JOIN    ${database}.${schema}.${demographics} d ON d.${patient_id} = c.pid
JOIN    _spans_all s ON s.pid = c.pid
GROUP BY c.pid, d.${birth_year}, d.${sex}, d.${zip3};

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
  SELECT m.${patient_id} AS pid, m.${claim_date} AS svc_date,
         'DX'  AS code_type, m.${icd_code} AS code, m.${place_of_service} AS pos
  FROM   ${database}.${schema}.${claims} m
  JOIN   cohort_ids c ON c.pid = m.${patient_id}
  WHERE  m.${claim_date} BETWEEN '${lookback_start}' AND '${end_date}'
    AND  m.${icd_code} IS NOT NULL
  UNION ALL
  -- procedures
  SELECT m.${patient_id}, m.${claim_date},
         'PROC', m.${procedure_code}, m.${place_of_service}
  FROM   ${database}.${schema}.${claims} m
  JOIN   cohort_ids c ON c.pid = m.${patient_id}
  WHERE  m.${claim_date} BETWEEN '${lookback_start}' AND '${end_date}'
    AND  m.${procedure_code} IS NOT NULL
  UNION ALL
  -- pharmacy (NDC) -- OPTIONAL: comment out this block to shrink the export
  SELECT p.${patient_id}, p.${fill_date},
         'RX', p.${ndc_code}, NULL
  FROM   ${database}.${schema}.${pharmacy} p
  JOIN   cohort_ids c ON c.pid = p.${patient_id}
  WHERE  p.${fill_date} BETWEEN '${lookback_start}' AND '${end_date}'
    AND  p.${ndc_code} IS NOT NULL
);

-- 4) arthritis events (outcome labels + epi numerator) ------------------------
-- Full ICD-10 codes used here (L40.5 needs 4 chars). One row/patient/subtype.
CREATE OR REPLACE TEMP TABLE _dx AS
SELECT  m.${patient_id} AS pid, m.${claim_date} AS dx_date,
        CASE
          WHEN m.${icd_code} LIKE 'M15%' OR m.${icd_code} LIKE 'M16%'
            OR m.${icd_code} LIKE 'M17%' OR m.${icd_code} LIKE 'M18%'
            OR m.${icd_code} LIKE 'M19%' THEN 'OA'
          WHEN m.${icd_code} LIKE 'M05%' OR m.${icd_code} LIKE 'M06%' THEN 'RA'
          WHEN m.${icd_code} LIKE 'M10%' THEN 'GOUT'
          WHEN m.${icd_code} LIKE 'L40.5%' OR m.${icd_code} LIKE 'M07%' THEN 'PSA'
          WHEN m.${icd_code} LIKE 'M45%' THEN 'AS'
          WHEN m.${icd_code} LIKE 'M08%' THEN 'JUV'
          WHEN m.${icd_code} LIKE 'M12%' OR m.${icd_code} LIKE 'M13%' THEN 'OTHER'
        END AS subtype
FROM    ${database}.${schema}.${claims} m
JOIN    cohort_ids c ON c.pid = m.${patient_id}
WHERE  (m.${icd_code} LIKE 'M0%' OR m.${icd_code} LIKE 'M1%'
     OR m.${icd_code} LIKE 'M45%' OR m.${icd_code} LIKE 'L40.5%');

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
           AND dd.dx_date >= DATEADD('day', ${second_claim_min_days}, f.first_dx)
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
