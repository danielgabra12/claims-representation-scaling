-- 01_extract.sql — THE SINGLE SNOWFLAKE EXTRACTION (templated).
-- =============================================================================
-- DO NOT edit names by hand here. This file contains <...>-style placeholders
-- that are rendered from config/schema_config.yaml by:  python sql/render.py
-- The rendered output (sql/01_extract.rendered.sql) is what you paste into
-- Snowflake.
--
-- One-pull principle (plan §1): pull RAW and WIDE. Exact birth years, full
-- enrollment spans, first + second qualifying diagnosis dates, claim counts,
-- baseline utilization. Every definitional choice (washout length, 1- vs
-- 2-claim case def, age banding) is then reversible LOCALLY in DuckDB.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table A — cohort / denominator (one row per eligible enrollment span).
-- Uses gaps-and-islands to convert monthly eligibility rows into spans,
-- treating gaps <= 45 days as continuous.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TEMP TABLE enroll_spans AS
WITH monthly AS (
  SELECT PAT_ID AS pid,
         MONTH_ID  AS month            -- DATE at month granularity
  FROM   PHARMETRICS.PUBLIC.ENROLLMENT
  -- Pull from lookback_start (not start_date) so the washout completes before
  -- 2019; the local pipeline restricts OBSERVED person-time to >= start_date.
  WHERE  MONTH_ID BETWEEN '2017-01-01' AND '2024-12-31'
),
flagged AS (
  SELECT pid, month,
         CASE WHEN DATEDIFF('day',
                LAG(month) OVER (PARTITION BY pid ORDER BY month),
                month) > 45
              THEN 1 ELSE 0 END AS new_span
  FROM monthly
),
grouped AS (
  SELECT pid, month,
         SUM(new_span) OVER (PARTITION BY pid ORDER BY month) AS span_id
  FROM flagged
)
SELECT pid, span_id,
       MIN(month) AS enroll_start,
       MAX(month) AS enroll_end
FROM grouped
GROUP BY pid, span_id;

CREATE OR REPLACE TEMP TABLE cohort AS
SELECT  s.pid,
        d.DOBYR AS birth_year,
        d.DER_SEX        AS sex,
        d.PAT_REGION     AS region,
        d.PAT_STATE      AS state,
        s.enroll_start,
        s.enroll_end,
        DATEDIFF('day', s.enroll_start, s.enroll_end) AS enrolled_days,
        -- pull-wide insurance for the optional ML stretch (cheap):
        (SELECT COUNT(*) FROM PHARMETRICS.PUBLIC.MEDICAL_CLAIMS c
           WHERE c.PAT_ID = s.pid
             AND c.FROM_DT BETWEEN s.enroll_start
                                     AND DATEADD('month', 12, s.enroll_start)
        ) AS baseline_claim_count
FROM    enroll_spans s
JOIN    PHARMETRICS.PUBLIC.PATIENT d
        ON d.PAT_ID = s.pid
WHERE   DATEDIFF('month', s.enroll_start, s.enroll_end) >= 12;

-- -----------------------------------------------------------------------------
-- Table B — events / numerator (one row per patient x subtype).
-- Captures first AND second qualifying diagnosis dates so BOTH the 1-claim and
-- 2-claim case definitions can be applied locally.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE TEMP TABLE dx AS
SELECT  PAT_ID AS pid,
        FROM_DT  AS dx_date,
        CASE
          WHEN DIAG LIKE 'M15%' OR DIAG LIKE 'M16%'
            OR DIAG LIKE 'M17%' OR DIAG LIKE 'M18%'
            OR DIAG LIKE 'M19%' THEN 'OA'
          WHEN DIAG LIKE 'M05%' OR DIAG LIKE 'M06%' THEN 'RA'
          WHEN DIAG LIKE 'M10%' THEN 'GOUT'
          WHEN DIAG LIKE 'L40.5%' OR DIAG LIKE 'M07%' THEN 'PSA'
          WHEN DIAG LIKE 'M45%' THEN 'AS'
          WHEN DIAG LIKE 'M08%' THEN 'JUV'
          WHEN DIAG LIKE 'M12%' OR DIAG LIKE 'M13%' THEN 'OTHER'
        END AS subtype
FROM    PHARMETRICS.PUBLIC.MEDICAL_CLAIMS
WHERE   DIAG LIKE 'M0%' OR DIAG LIKE 'M1%'
     OR DIAG LIKE 'M45%' OR DIAG LIKE 'L40.5%';

CREATE OR REPLACE TEMP TABLE events AS
WITH distinct_days AS (
  SELECT DISTINCT pid, subtype, dx_date FROM dx WHERE subtype IS NOT NULL
),
firsts AS (
  SELECT pid, subtype, MIN(dx_date) AS first_dx,
         COUNT(*) AS n_claim_days
  FROM distinct_days GROUP BY pid, subtype
)
SELECT f.pid, f.subtype, f.first_dx, f.n_claim_days,
       (SELECT MIN(dd.dx_date) FROM distinct_days dd
         WHERE dd.pid = f.pid AND dd.subtype = f.subtype
           AND dd.dx_date >= DATEADD('day', 30, f.first_dx)
       ) AS second_qualifying_dx          -- NULL if no 2nd claim >= 30d later
FROM firsts f;

-- -----------------------------------------------------------------------------
-- QC — run and EYEBALL before exporting (plan §2 step 4). Never download blind.
-- -----------------------------------------------------------------------------
SELECT 'cohort_n'        AS metric, COUNT(*)::STRING AS value FROM cohort
UNION ALL SELECT 'birth_year_min', MIN(birth_year)::STRING FROM cohort
UNION ALL SELECT 'birth_year_max', MAX(birth_year)::STRING FROM cohort;
SELECT sex,     COUNT(*) AS n FROM cohort GROUP BY sex     ORDER BY sex;
SELECT region,  COUNT(*) AS n FROM cohort GROUP BY region  ORDER BY region;  -- all 4?
SELECT subtype, COUNT(*) AS n FROM events GROUP BY subtype ORDER BY subtype; -- plausible?

-- -----------------------------------------------------------------------------
-- Export — parquet preferred (size). Adjust the local path in the GET commands.
-- -----------------------------------------------------------------------------
COPY INTO @~/surf/cohort FROM cohort
  FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE;
COPY INTO @~/surf/events FROM events
  FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE;
-- GET @~/surf/cohort file:///C:/Users/Daniel/OneDrive/Desktop/SURF%20Project/data/raw/;
-- GET @~/surf/events file:///C:/Users/Daniel/OneDrive/Desktop/SURF%20Project/data/raw/;
-- (Or use the Snowflake UI "Download" on SELECT * FROM cohort / events.)
