# Stage 6 — Arm A (GBDT) across the scaling ladder: Arthritis (any subtype)

_Engineered features -> XGBoost. Fixed temporal test set. Mean +/- SD over 3 seeds. Three feature sets (see reports/fairness_contract.md): `thin` (original 27), `rich_full` (fair baseline, all families), `rich_noutil` (rich minus utilization+temporal = access-confound ablation)._

## Learning curve (temporal holdout, AUROC mean ± SD)
| size | thin_full | rich_full | rich_noutil | Δ richness | Δ util |
|---|---|---|---|---|---|
| 10,000 | 0.7974 ± 0.0036 | 0.7865 ± 0.0098 | 0.7814 ± 0.0105 | -0.0109 | +0.0052 |
| 30,000 | 0.8061 ± 0.0042 | 0.8094 ± 0.0092 | 0.8134 ± 0.0037 | +0.0033 | -0.0040 |
| 100,000 | 0.8142 ± 0.0028 | 0.8177 ± 0.0011 | 0.8174 ± 0.0008 | +0.0035 | +0.0004 |
| 300,000 | 0.8189 ± 0.0006 | 0.8230 ± 0.0007 | 0.8216 ± 0.0003 | +0.0040 | +0.0014 |
| 1,000,000 | 0.8234 ± 0.0014 | 0.8272 ± 0.0013 | 0.8255 ± 0.0005 | +0.0038 | +0.0017 |
| 1,114,144 | 0.8239 ± 0.0002 | 0.8279 ± 0.0005 | 0.8256 ± 0.0006 | +0.0040 | +0.0022 |

- **Δ richness** = rich_full − thin_full: how much the bag-of-codes / drug-class enrichment bought (the fairness upgrade).
- **Δ util** = rich_full − rich_noutil: the portion attributable to the healthcare-access confound rather than clinical content.

## AUPRC / calibration (Brier), rich_full
| size | AUPRC | Brier |
|---|---|---|
| 10,000 | 0.0601 | 0.0180 |
| 30,000 | 0.0712 | 0.0178 |
| 100,000 | 0.0794 | 0.0176 |
| 300,000 | 0.0844 | 0.0175 |
| 1,000,000 | 0.0887 | 0.0175 |
| 1,114,144 | 0.0891 | 0.0175 |

## Reading it
- At the largest local size (1,114,144): thin **0.824** -> rich_full **0.828** (Δ richness +0.004). Regression guard vs thin baseline: **PASS**.
- Removing utilization+temporal from rich drops it to **0.826** (Δ util +0.002) — the access-confound share.
- `rich_full` is the Arm A half of the crossover plot; the Arm B curve (Stage 7) overlays on the same axes. Local sizes stop at the train-pool max; >=1M needs the full extraction.

Per-run configs/metrics: `experiments/arthritis/armA/`. Raw table: `reports/stage6_arthritis_armA_results.csv`.