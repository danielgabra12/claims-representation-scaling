# Stage 7 — Arm B (1D CNN) and the A-vs-B crossover: Arthritis (any subtype)

_Ordered code sequences -> 1D CNN (Deepr-style). Same ladder, val split, and fixed temporal test set as Arm A; identical information, different representation. Mean ± SD over 3 seeds._

## Learning curve (temporal holdout, AUROC mean ± SD)
| size | Arm A (rich GBDT) | Arm B (1D CNN) | Δ (B − A) |
|---|---|---|---|
| 10,000 | 0.7865 ± 0.0098 | 0.7160 ± 0.0033 | -0.0705 |
| 30,000 | 0.8094 ± 0.0092 | 0.7399 ± 0.0031 | -0.0695 |
| 100,000 | 0.8177 ± 0.0011 | 0.7565 ± 0.0009 | -0.0612 |
| 300,000 | 0.8230 ± 0.0007 | 0.7772 ± 0.0045 | -0.0458 |
| 1,000,000 | 0.8272 ± 0.0013 | 0.7939 ± 0.0016 | -0.0333 |
| 1,114,144 | 0.8279 ± 0.0005 | 0.7920 ± 0.0042 | -0.0358 |

## AUPRC / calibration (Brier), Arm B
| size | AUPRC | Brier |
|---|---|---|
| 10,000 | 0.0416 | 0.0180 |
| 30,000 | 0.0471 | 0.0180 |
| 100,000 | 0.0534 | 0.0182 |
| 300,000 | 0.0638 | 0.0180 |
| 1,000,000 | 0.0718 | 0.0178 |
| 1,114,144 | 0.0702 | 0.0177 |

## Reading it
- At the largest local size (1,114,144): Arm A **0.828** vs Arm B **0.792** (Δ -0.036). Arm B is closing.
- Whether the curves cross depends on scale beyond the local sample; the ≥1M rungs need the full extraction (the study's #1 gate). The **slope** of Δ across the ladder is the local signal for whether a crossover is coming.

Overlay figure: `reports/figures/stage7_arthritis_crossover.png`. Raw table: `reports/stage7_arthritis_armB_results.csv`.