# Stage 3 — Engineered features: Arthritis (any subtype)

_Arm A inputs. Every feature from the baseline window only (strictly pre-index); leakage audit passed. Utilization features tagged for the Stage 6 ablation._

- **Rows:** 595,758 (one per cohort patient). **Features:** 27 = 8 utilization + 19 clinical/demographic.
- **No-baseline-claim patients:** 94,421 (enrolled but zero utilization; kept, features=0).
- **Mean baseline utilization:** 48.3 claims, 18.6 dx, 5.39 active months.

## Comorbidity prevalence, and by outcome (signal check)
A predictor should be more prevalent among positives than negatives.

| comorbidity | overall % | negatives % | positives % |
|---|---|---|---|
| cm_htn | 20.9 | 20.3 | 50.0 |
| cm_lipid | 20.3 | 19.8 | 46.4 |
| cm_diabetes | 8.5 | 8.2 | 20.9 |
| cm_obesity | 10.7 | 10.4 | 20.9 |
| cm_depr_anx | 15.9 | 15.7 | 23.5 |
| cm_ckd | 2.0 | 1.9 | 6.7 |
| cm_ihd | 3.5 | 3.4 | 10.6 |
| cm_osteoporosis | 1.4 | 1.3 | 4.0 |
| cm_backpain | 13.8 | 13.5 | 29.5 |
| cm_jointpain | 10.9 | 10.5 | 28.9 |
| cm_soft_msk | 12.8 | 12.4 | 30.9 |
| cm_tobacco | 9.6 | 9.5 | 16.7 |

## Utilization features (removed in the ablation arm)
`n_claims`, `n_dx`, `n_proc`, `n_rx`, `n_dx_codes`, `n_ndc`, `n_proc_codes`, `n_active_months`

Table: `data/processed/arthritis_features.parquet` (gitignored). Column tags: `data/processed/arthritis_features_meta.json`.