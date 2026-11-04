# Stage 8 — Arm C (Transformer) three-way comparison: Arthritis (any subtype)

_Self-attention encoder (BEHRT-style) over the same tokens/vocab/ladder/val/test/seeds as Arm A (GBDT) and Arm B (CNN). Fixed documented config. Mean ± SD over 3 seeds._

## Learning curves (temporal holdout, AUROC mean ± SD)
| size | Arm A (GBDT) | Arm B (CNN) | Arm C (Transformer) |
|---|---|---|---|
| 10,000 | 0.7970 | 0.7158 | 0.6163 ± 0.0900 |
| 30,000 | 0.8079 | 0.7362 | 0.7449 ± 0.0039 |
| 100,000 | 0.8181 | 0.7571 | 0.7666 ± 0.0006 |
| 300,000 | 0.8231 | 0.7734 | 0.7822 ± 0.0005 |
| 1,000,000 | 0.8269 | 0.7929 | 0.8057 ± 0.0018 |
| 1,114,144 | 0.8278 | 0.7953 | 0.8063 ± 0.0006 |

## Reading it
- At the largest local size (1,114,144): Arm A **0.828**, Arm B (CNN) **0.795**, Arm C (Transformer) **0.806**.
- The modern Transformer **beats** the 2017 CNN here (Δ +0.011); vs engineered features the gap is -0.021. This directly answers the 'did you try a Transformer?' question.

Overlay: `reports/figures/stage8_arthritis_threeway.png`. Raw: `reports/stage8_arthritis_armC_results.csv`.