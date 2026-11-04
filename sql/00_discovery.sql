-- 00_discovery.sql — schema discovery (Gate A helper).
-- CHEAP, read-only. Run this FIRST, in the same Snowflake session as the
-- extraction, to learn IQVIA's real table/column names. Paste the output
-- back to Claude (or into config/schema_config.yaml).
--
-- Goal: identify three tables —
--   (a) enrollment / eligibility   (one row per patient-month, or span rows)
--   (b) diagnosis / claims         (claim-level ICD-10 rows)
--   (c) member demographics        (birth year, sex, region/state)

-- 1) What databases can you see?
SHOW DATABASES;

-- 2) Largest tables in the candidate database (replace <DATABASE>):
SELECT TABLE_SCHEMA, TABLE_NAME, ROW_COUNT
FROM   <DATABASE>.INFORMATION_SCHEMA.TABLES
ORDER  BY ROW_COUNT DESC
LIMIT  50;

-- 3) Columns for each candidate table (run once per table, replace <TABLE>):
SELECT COLUMN_NAME, DATA_TYPE
FROM   <DATABASE>.INFORMATION_SCHEMA.COLUMNS
WHERE  TABLE_NAME = '<TABLE>'
ORDER  BY ORDINAL_POSITION;

-- 4) Peek at a few rows to confirm meaning (replace <DATABASE>.<SCHEMA>.<TABLE>):
SELECT * FROM <DATABASE>.<SCHEMA>.<TABLE> LIMIT 20;

-- KEY QUESTION to answer from the enrollment table:
--   Is enrollment stored as MONTHLY rows (one row per patient per month)
--   or as SPANS (explicit start/end dates)?  This sets enrollment_storage
--   in config/schema_config.yaml and which branch of 01_extract.sql renders.
