"""Single source of truth for the data-scale crossover study.

Every stage imports its constants from here. Nothing in this file should require
touching Snowflake; the local Parquet sample is the substrate. Values that are
still scientific judgment calls (treatment sets, final cohort rules) are marked
PENDING and must be resolved with the PI (Cyril sign-off) before Stage 2, not
silently chosen here.
"""
from __future__ import annotations

from pathlib import Path

# --- Reproducibility ---------------------------------------------------------
# One global seed governs every stochastic choice (subsample draws, model init,
# shuffles). Per-run seeds in the scaling ladder are derived as GLOBAL_SEED + k.
GLOBAL_SEED = 20260705  # project date; recorded in every config.json

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"           # immutable Parquet sample; never written
DATA_PROCESSED = ROOT / "data" / "processed"  # per-disease cohorts/labels/features/seqs
DATA_SPLITS = ROOT / "data" / "splits"        # persisted patient-level partitions + ladder
EXPERIMENTS = ROOT / "experiments"            # one dir per (disease,arm,size,seed)
REPORTS = ROOT / "reports"

for _d in (DATA_PROCESSED, DATA_SPLITS, EXPERIMENTS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Raw local data ----------------------------------------------------------
# SCALE PULL (2026-07-16): 20% sample of eligible patients from the live IQVIA share
# (see sql/EXTRACT_next_session.sql). Multi-part Parquet (64 parts each) read via glob.
#   patients   2,017,387 | enrollment 94,078,458 | claims 1,278,651,066 rows.
# The earlier MOD(HASH,10) ~2.2M "*_2m" sample is superseded (kept until disk needs it).
RAW = {
    "claims":     DATA_RAW / "claims_*.snappy.parquet",
    "patients":   DATA_RAW / "patients_*.snappy.parquet",
    "enrollment": DATA_RAW / "enrollment_*.snappy.parquet",
    "dx_lookup":  DATA_RAW / "dx_lookup.parquet",
    "rx_lookup":  DATA_RAW / "rx_lookup.parquet",
    "pr_lookup":  DATA_RAW / "pr_lookup.parquet",
}

# Verified schema facts the pipeline relies on (see reports/stage0_data_manifest.md):
#   claims:   PID (== patients.PAT_ID, 1:1, no orphans), MONTH_ID (int YYYYMM),
#             CODE_TYPE in {DX, PROC, RX}, CODE (raw code string, no dots)
#   patients: PAT_ID, FIRST_MONTH, LAST_MONTH, N_ENROLLED_MONTHS, DER_YOB, DER_SEX,
#             PAT_STATE, PAT_ZIP3, PAT_REGION, DOD
#   enrollment: PAT_ID, MONTH_ID, PAY_TYPE
PATIENT_KEY = "PAT_ID"     # canonical patient id used everywhere downstream
CLAIMS_KEY = "PID"         # same identifier as PAT_ID in the claims table
CODE_TYPES = ("DX", "PROC", "RX")

# Data-quality guards to enforce at the cohort stage (Stage 2), recorded now:
#   DER_YOB spans 0..2025  -> 0 is missing; treat implausible YOB as excluded.
#   DER_SEX in {F, M, U}   -> U (~37.6k) is unknown.
#   PAT_REGION in {MW,NE,S,W,None} -> ~54.3k null region.
YOB_MIN_VALID = 1900       # exclude DER_YOB < this (catches 0) at cohort build
YOB_MAX_VALID = 2025       # newborns exist; cohort age filter handles the rest

# --- Study calendar ----------------------------------------------------------
# MONTH_ID is an int YYYYMM. Local sample spans 201901..202504.
DATA_MONTH_MIN = 201901
DATA_MONTH_MAX = 202504    # 2025 is a PARTIAL year -> excluded from temporal holdout

# --- Temporal holdout (FINALIZED at Stage 2) ---------------------------------
# Index date = per-patient FIRST_MONTH + 12, so index years are bucketed by when
# a patient's enrollment starts. With BASELINE_MONTHS=12 + PREDICTION_WINDOW=12 =
# 24 months per patient, and enrollment data ending 202502, the full index-years
# that support a completely-observed outcome window are 2020-2023. Train on the
# earlier full years, test on the latest full year (2023).
TRAIN_INDEX_YEARS = (2020, 2021, 2022)  # train-eligible index years
TEST_INDEX_YEAR = 2023                   # held-out latest full index year
AGE_MIN, AGE_MAX = 18, 89                # adult at index; upper cap drops bad YOB

# --- Cohort / labeling windows (FINALIZED at Stage 2) ------------------------
BASELINE_MONTHS = 12          # pre-index observable history -> defines index date
# W=12 not 24: the 76-month span (enrollment ends 202502) cannot leave a FULL
# held-out test year at W=24. W=24 survives only as a landmark-design sensitivity.
PREDICTION_WINDOW_MONTHS = 12  # outcome must fall within this window after index
PHENOTYPE_MIN_GAP_DAYS = 30   # "two qualifying dx >=30 days apart" (approx via months)

# --- Scaling ladder (spec Section 6.2) ---------------------------------------
# Geometric training-size ladder, >=6 points spanning >=3 orders of magnitude.
# LOCAL sample caps the realizable top: ~1.89M patients HAVE claims, and fewer
# survive cohort restriction, so 3M/10M points require the full-population
# extraction (Stage 1, human-only). "max" = all eligible training patients.
SCALING_LADDER = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]  # + "max"
N_SEEDS_PER_POINT = 3         # >=3 seeds/point; mean +/- SD, error bars mandatory

# --- Rich Arm A feature config (Stage 3.1; see reports/fairness_contract.md) --
# Bag-of-codes vocabularies are chosen LABEL-FREE from the TRAIN POOL only (top-K by
# patient frequency), frozen, then applied to every ladder size, seed, and the test set.
RICH_DX_VOCAB_K = 500     # top DX 3-char ICD-10 categories
RICH_PROC_VOCAB_K = 200   # top procedure codes
# Disease-specific treatments dropped from the rx_class block (decision #1: keep
# symptom/analgesic signal, drop disease-specific treatment that encodes uncoded disease).
# Two granularities, applied together (PENDING Cyril sign-off; documented + overridable):
#   *_USC   : drop specific rx_lookup.USC_NAME groups. Used when the therapeutic CLASS
#             mixes drop-these with keep-these drugs (arthritis: ANTIARTHRITICS class holds
#             both disease-specific DMARDs to drop and NSAIDs/COX-2 to keep).
#   *_CLASS : drop a whole rx_lookup.THPTC_CLAS_DESC. Used when the class is entirely
#             disease-specific (t2d: DIABETIC THERAPY = all 34 antidiabetic drug groups).
RX_TREATMENT_EXCLUDE_USC = {
    "arthritis": ["ANTIARTH, DMARDS",
                  "ANTIARTH, BIOLOGICAL RESPONSE MODIFIERS",
                  "ANTIARTH, GOUT SPECIFIC"],
    "t2d": [],
    "heart_failure": [],  # to define at phenotype sign-off
}
RX_TREATMENT_EXCLUDE_CLASS = {
    "arthritis": [],
    "t2d": ["DIABETIC THERAPY"],   # all antidiabetics; supplies/monitoring deferred
    "heart_failure": [],           # HF drugs overlap hypertension -> needs care; PENDING
}

# --- Disease registry --------------------------------------------------------
# Build the pipeline once, run across >=2 (target 3) diseases (spec Section 5).
# `dx_regex` matches the pipe-concatenated diagnosis string (codes have no dots).
# `treatment` (drugs/procedures) is the confound-hardening half of the phenotype
# and is a scientific judgment call -> PENDING PI/Cyril sign-off before Stage 2.
DISEASES = {
    "arthritis": {
        # Validated in the descriptive report against CDC NHIS 2022.
        "label": "Arthritis (any subtype)",
        "dx_regex": r"^(M0[5-8]|M1[0235-9]|M45|L405)",  # applied per DX code
        "treatment": None,  # PENDING: DMARDs/biologics/urate-lowering/intra-articular etc.
        "base_rate_hint": 0.075,  # ~7.5% onset at the 2-visit/24-mo design (handoff)
    },
    # Candidate disease #2 / #3 (differing base rates; PI selects):
    "t2d": {
        "label": "Type 2 diabetes",
        "dx_regex": r"^E11",
        "treatment": None,  # PENDING: metformin / insulin / other antihyperglycemics
        "base_rate_hint": None,
    },
    "heart_failure": {
        "label": "Heart failure",
        "dx_regex": r"^I50",
        "treatment": None,  # PENDING: loop diuretics / ARNI / beta-blocker context
        "base_rate_hint": None,
    },
}
PRIMARY_DISEASE = "arthritis"


def seed_for(k: int) -> int:
    """Derive the k-th run seed deterministically from the global seed."""
    return GLOBAL_SEED + k


def mem_limit_gb(headroom_gb: float = 2.0, floor: int = 3, cap: int = 8) -> int:
    """A DuckDB memory_limit that fits current free RAM (this is a 16GB box whose free
    RAM swings a lot). Leaves headroom for pandas; falls back to `floor` without psutil.
    Setting the limit below available RAM makes DuckDB spill to temp instead of OOMing."""
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        return int(max(floor, min(cap, avail - headroom_gb)))
    except Exception:
        return floor
