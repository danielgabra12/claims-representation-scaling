# Stage 3.1 — Enriched features: Arthritis (any subtype)

_Fair-baseline Arm A inputs (see reports/fairness_contract.md). Bags chosen label-free from the train pool, frozen. Leakage + phenotype-exclusion audits passed._

- **Rows:** 1,251,833. **Features:** 828 across 7 families.
- **No-baseline-claim patients:** 196,734 (kept, features=0).
- **RX claims dropped as disease-specific treatment:** 50,433 (USC in ['ANTIARTH, DMARDS', 'ANTIARTH, BIOLOGICAL RESPONSE MODIFIERS', 'ANTIARTH, GOUT SPECIFIC']).

## Feature families
| family | count | in access-confound ablation |
|---|---|---|
| demo | 7 | kept |
| comorbidity | 12 | kept |
| utilization | 8 | REMOVED |
| temporal | 5 | REMOVED |
| dx_bag | 500 | kept |
| rx_class | 96 | kept |
| proc_bag | 200 | kept |

## Derived column sets (Stage 6 selects on these)
- **full** (rich): 828  ·  **clinical** (no-util): 815  ·  **thin** (original 27): 27

## Signal check — mean count by outcome (top new features)
| feature | negatives | positives |
|---|---|---|
| dx_z00 | 1.879 | 2.462 |
| dx_z23 | 1.039 | 1.571 |
| dx_z12 | 1.193 | 2.332 |
| dx_i10 | 2.781 | 8.48 |
| dx_e78 | 2.257 | 6.468 |
| rx_biologicals | 0.271 | 0.388 |
| rx_glucocorticoids | 0.297 | 0.646 |
| rx_psychostimulants_antidepressants | 0.866 | 1.536 |
| rx_antiarthritics | 0.255 | 0.698 |
| rx_other_antihypertensives | 0.63 | 1.462 |

Table: `data/processed/arthritis_features.parquet` (gitignored). Tags: `arthritis_features_meta.json`. Frozen vocab: `arthritis_features_vocab.json`.