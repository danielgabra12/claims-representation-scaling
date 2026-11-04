-- =============================================================================
-- THE SINGLE IQVIA EXTRACTION — read-only-role edition (no temp tables).
-- Confirmed schema: HDSC_CHAPMAN_PROD.INTG_CDWHD_PHAR (2026-07-16).
--
-- Your role HDSC_CHAPMAN_PROD_RO can SELECT the share and write to your own user
-- stage (@~) but cannot CREATE anything. So this uses NO temp tables: the cohort
-- logic is a CTE repeated inside each statement, and results stream straight to
-- Parquet on your user stage. The repeated enrollment scan is cheap on your LARGE
-- warehouse.
--
-- HOW TO RUN:
--   1. Run STEP 1 (QC) first. Send me the result. It must show cohort ~>=1,000,000
--      and first_month between ~201801 and ~202301.
--   2. If it looks right, run STEP 2 (the three COPY exports).
--   3. Download the three stage folders into data/raw/ (see GET lines), reply "snowflake done".
--
-- Cohort definition (same as the local pipeline): patients with a >=24-month
-- continuous enrollment span (no skipped months), 20% hash sample (~2.0M patients;
-- 40% overshot to 4.0M at QC). FIRST_MONTH =
-- start of their earliest qualifying span. Claims are unpivoted from IQVIA-wide
-- (DIAG1..DIAG12) into the long (PID, MONTH_ID, CODE_TYPE, CODE) stream.
-- =============================================================================

-- ============================ STEP 1 — QC (run first) ========================
WITH sp AS (
  SELECT e.PAT_ID AS pid, e.MONTH AS ym,
         (FLOOR(e.MONTH/100)*12 + MOD(e.MONTH,100) - 1) AS mabs
  FROM   HDSC_CHAPMAN_PROD.INTG_CDWHD_PHAR.PHAR_ENROLL2_DIM e
  WHERE  e.MONTH BETWEEN 201701 AND 202412
),
fl AS (
  SELECT pid, ym, mabs,
         CASE WHEN mabs - LAG(mabs) OVER (PARTITION BY pid ORDER BY mabs) > 1
              THEN 1 ELSE 0 END AS nw
  FROM sp
),
gr AS (
  SELECT pid, ym, mabs, SUM(nw) OVER (PARTITION BY pid ORDER BY mabs) AS sid FROM fl
),
spans AS (
  SELECT pid, sid, MIN(ym) AS es, MAX(ym) AS ee, (MAX(mabs)-MIN(mabs)+1) AS sm
  FROM gr GROUP BY pid, sid
),
qual AS (
  SELECT pid, es, ee, sm FROM (
    SELECT pid, es, ee, sm, ROW_NUMBER() OVER (PARTITION BY pid ORDER BY es) rn
    FROM spans WHERE sm >= 24 AND MOD(ABS(HASH(pid)),100) < 20
  ) WHERE rn = 1
)
SELECT COUNT(*)   AS cohort_patients,
       MIN(es)    AS first_month_min,
       MAX(es)    AS first_month_max,
       MIN(sm)    AS min_span_months
FROM qual;

-- ============================ STEP 2 — EXPORTS ===============================
-- (Run these three only after the QC above looks right.)

-- 2a) patients ---------------------------------------------------------------
COPY INTO @~/iqvia/patients FROM (
  WITH sp AS (
    SELECT e.PAT_ID AS pid, e.MONTH AS ym,
           (FLOOR(e.MONTH/100)*12 + MOD(e.MONTH,100) - 1) AS mabs
    FROM HDSC_CHAPMAN_PROD.INTG_CDWHD_PHAR.PHAR_ENROLL2_DIM e
    WHERE e.MONTH BETWEEN 201701 AND 202412),
  fl AS (SELECT pid, ym, mabs,
           CASE WHEN mabs - LAG(mabs) OVER (PARTITION BY pid ORDER BY mabs) > 1
                THEN 1 ELSE 0 END AS nw FROM sp),
  gr AS (SELECT pid, ym, mabs, SUM(nw) OVER (PARTITION BY pid ORDER BY mabs) AS sid FROM fl),
  spans AS (SELECT pid, sid, MIN(ym) es, MAX(ym) ee, (MAX(mabs)-MIN(mabs)+1) sm
            FROM gr GROUP BY pid, sid),
  qual AS (SELECT pid, es, ee, sm FROM (
             SELECT pid, es, ee, sm, ROW_NUMBER() OVER (PARTITION BY pid ORDER BY es) rn
             FROM spans WHERE sm >= 24 AND MOD(ABS(HASH(pid)),100) < 20) WHERE rn = 1)
  SELECT q.pid AS PAT_ID, q.es AS FIRST_MONTH, q.ee AS LAST_MONTH,
         q.sm AS N_ENROLLED_MONTHS,
         d.DER_YOB, d.DER_SEX, d.PAT_STATE, d.PAT_ZIP3, d.PAT_REGION, d.DOD
  FROM qual q
  JOIN (SELECT PATIENT_ID, DERIVED_BIRTH_YEAR_NBR AS DER_YOB, DER_SEX,
               PAT_STATE, PAT_ZIP3, PAT_REGION, DOD
        FROM HDSC_CHAPMAN_PROD.INTG_CDWHD_PHAR.PHAR_ENROLL_SYNTH_DIM
        QUALIFY ROW_NUMBER() OVER (PARTITION BY PATIENT_ID ORDER BY DERIVED_BIRTH_YEAR_NBR)=1) d
    ON d.PATIENT_ID = q.pid
) FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE;

-- 2b) enrollment (monthly rows for the cohort) -------------------------------
COPY INTO @~/iqvia/enrollment FROM (
  WITH sp AS (
    SELECT e.PAT_ID AS pid, e.MONTH AS ym,
           (FLOOR(e.MONTH/100)*12 + MOD(e.MONTH,100) - 1) AS mabs
    FROM HDSC_CHAPMAN_PROD.INTG_CDWHD_PHAR.PHAR_ENROLL2_DIM e
    WHERE e.MONTH BETWEEN 201701 AND 202412),
  fl AS (SELECT pid, ym, mabs,
           CASE WHEN mabs - LAG(mabs) OVER (PARTITION BY pid ORDER BY mabs) > 1
                THEN 1 ELSE 0 END AS nw FROM sp),
  gr AS (SELECT pid, ym, mabs, SUM(nw) OVER (PARTITION BY pid ORDER BY mabs) AS sid FROM fl),
  spans AS (SELECT pid, sid, MIN(ym) es, MAX(ym) ee, (MAX(mabs)-MIN(mabs)+1) sm
            FROM gr GROUP BY pid, sid),
  qual AS (SELECT pid FROM (
             SELECT pid, ROW_NUMBER() OVER (PARTITION BY pid ORDER BY es) rn
             FROM spans WHERE sm >= 24 AND MOD(ABS(HASH(pid)),100) < 20) WHERE rn = 1)
  SELECT e.PAT_ID, e.MONTH AS MONTH_ID, e.PAY_TYPE
  FROM HDSC_CHAPMAN_PROD.INTG_CDWHD_PHAR.PHAR_ENROLL2_DIM e
  WHERE e.MONTH BETWEEN 201701 AND 202412
    AND e.PAT_ID IN (SELECT pid FROM qual)
) FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE;

-- 2c) claims (wide -> long code stream) --------------------------------------
COPY INTO @~/iqvia/claims FROM (
  WITH sp AS (
    SELECT e.PAT_ID AS pid, e.MONTH AS ym,
           (FLOOR(e.MONTH/100)*12 + MOD(e.MONTH,100) - 1) AS mabs
    FROM HDSC_CHAPMAN_PROD.INTG_CDWHD_PHAR.PHAR_ENROLL2_DIM e
    WHERE e.MONTH BETWEEN 201701 AND 202412),
  fl AS (SELECT pid, ym, mabs,
           CASE WHEN mabs - LAG(mabs) OVER (PARTITION BY pid ORDER BY mabs) > 1
                THEN 1 ELSE 0 END AS nw FROM sp),
  gr AS (SELECT pid, ym, mabs, SUM(nw) OVER (PARTITION BY pid ORDER BY mabs) AS sid FROM fl),
  spans AS (SELECT pid, sid, (MAX(mabs)-MIN(mabs)+1) sm FROM gr GROUP BY pid, sid),
  qual AS (SELECT DISTINCT pid FROM spans WHERE sm >= 24 AND MOD(ABS(HASH(pid)),100) < 20),
  cc AS (
    SELECT m.PAT_ID AS PID, m.MONTH AS MONTH_ID,
           m.DIAG1, m.DIAG2, m.DIAG3, m.DIAG4, m.DIAG5, m.DIAG6,
           m.DIAG7, m.DIAG8, m.DIAG9, m.DIAG10, m.DIAG11, m.DIAG12,
           m.PROC_CDE, m.NDC
    FROM HDSC_CHAPMAN_PROD.INTG_CDWHD_PHAR.PHAR_CLAIMS_FACT m
    WHERE m.MONTH BETWEEN 201701 AND 202412
      AND m.PAT_ID IN (SELECT pid FROM qual))
  SELECT PID, MONTH_ID, 'DX' AS CODE_TYPE, UPPER(TRIM(code)) AS CODE
  FROM cc UNPIVOT (code FOR src IN (DIAG1,DIAG2,DIAG3,DIAG4,DIAG5,DIAG6,
                                    DIAG7,DIAG8,DIAG9,DIAG10,DIAG11,DIAG12))
  WHERE TRIM(code) <> ''
  UNION ALL
  SELECT PID, MONTH_ID, 'PROC', UPPER(TRIM(PROC_CDE)) FROM cc
    WHERE PROC_CDE IS NOT NULL AND TRIM(PROC_CDE) <> ''
  UNION ALL
  SELECT PID, MONTH_ID, 'RX', UPPER(TRIM(NDC)) FROM cc
    WHERE NDC IS NOT NULL AND TRIM(NDC) <> ''
) FILE_FORMAT=(TYPE=PARQUET) HEADER=TRUE OVERWRITE=TRUE MAX_FILE_SIZE=268435456;

-- ============================ STEP 3 — DOWNLOAD ==============================
-- In a local terminal with SnowSQL (or the Snowsight stage UI), pull each folder:
--   GET @~/iqvia/patients   file://<repo>/data/raw/;
--   GET @~/iqvia/enrollment file://<repo>/data/raw/;
--   GET @~/iqvia/claims     file://<repo>/data/raw/;
-- claims arrives as several .parquet parts — keep them all in data/raw/.
-- dx_lookup / pr_lookup / rx_lookup already exist locally — leave them.
