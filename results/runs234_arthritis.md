# Runs 2, 3 and 4: schedule, cohort sensitivity, and the corrected 10k rung

_68 runs from the 2026-08-19 GPU session, all integrity-checked against the 137,689-patient holdout (exact PAT_ID alignment, 2,554 cases, AUROC recomputing to metrics.json). Zero failures._

## Run 2: longer optimisation schedule

Arms B and C: epoch cap 25 to 100, patience 4 to 15, cosine decay added. Arm A: matched extension, 800 to 2,400 rounds and patience 40 to 100.

| arm | rung | frozen | extended | delta | best ep frozen | best ep extended | cap |
|---|---|---|---|---|---|---|---|
| A | 300,000 | 0.8230 | 0.8230 | +0.0000 | 137.0 | 137.0 | 2,400 |
| A | 1,114,144 | 0.8279 | 0.8279 | +0.0000 | 267.7 | 267.7 | 2,400 |
| B | 300,000 | 0.7772 | 0.7758 | -0.0014 | 1.7 | 2.7 | 100 |
| B | 1,114,144 | 0.7920 | 0.7929 | +0.0009 | 3.0 | 3.0 | 100 |
| C | 300,000 | 0.7822 | 0.7821 | -0.0000 | 3.7 | 3.7 | 100 |
| C | 1,114,144 | 0.8063 | 0.8064 | +0.0001 | 6.3 | 6.3 | 100 |

No arm moves by more than 0.0014 and **the selected epoch is identical to the frozen-budget run**. Arm C stops at epoch 6 of 100 allowed; Arm A uses 268 of 2,400 rounds. The models are not budget-limited: they stop because validation performance stops improving. This refutes the earlier reading that optimisation budget was a binding constraint.

## Run 4: corrected 10k Transformer

| config | AUROC | SD | best epoch | seeds |
|---|---|---|---|---|
| frozen (warmup 300, patience 4) | 0.6163 | 0.0900 | 4.3 | 3 |
| corrected (warmup 100, patience 10) | **0.7115** | 0.0302 | 12.4 | 5 |

Per-seed corrected: 0.6587, 0.7163, 0.7218, 0.7283, 0.7326. C minus B at 10k moves from -0.0997 to -0.0045, so the Transformer is close to level with the 2017 CNN there rather than far behind.

## Run 3: index-2020 sensitivity

Training pool restricted to index years 2021-2022: **424,458** patients, from 1,114,144, removing all 597,065 enrollment-boundary patients. Same fixed 2023 test set.

| rung | A | B | C | A-C restricted | A-C full pool |
|---|---|---|---|---|---|
| 10,000 | 0.7938 | 0.6911 | 0.6642 | +0.1297 | +0.1703 |
| 30,000 | 0.8098 | 0.7383 | 0.7450 | +0.0648 | +0.0645 |
| 100,000 | 0.8165 | 0.7480 | 0.7715 | +0.0450 | +0.0511 |
| 300,000 | 0.8227 | 0.7753 | 0.7829 | +0.0398 | +0.0408 |
| 424,458 | 0.8250 | 0.7838 | 0.7869 | +0.0381 | n/a |

At matched training size the gaps are essentially unchanged, so the enrollment-boundary confound affects the absolute level of every arm but does not produce the arm ranking or the shape of the curves.

_Arm C at 10k in Run 3 used the frozen configuration and shows the same training failure documented in Run 4, which is why that row is low._

## Run 7: is schedule inertness demonstrated where it is invoked?

Section 4.4 promotes the parity figure from a bound to an estimate on the strength of the 10k
correction, and defends that correction's mixed configuration by pointing at Run 2. But Run 2
only tested 300k and 1,114,144, so the argument did not cover the rungs between. Re-running Arm
C at 30k and 100k under the **corrected** configuration (warmup 100, patience 10) closes that
gap.

| rung | frozen | corrected | dAUROC | frozen AUPRC | corrected AUPRC | dAUPRC | best ep frozen | best ep corrected |
|---|---|---|---|---|---|---|---|---|
| 30,000 | 0.7449 | 0.7469 | +0.0020 | 0.0530 | 0.0536 | +0.0006 | 7.7 | 6.3 |
| 100,000 | 0.7666 | 0.7674 | +0.0008 | 0.0597 | 0.0599 | +0.0002 | 3.3 | 2.7 |

Both deltas are within 0.002 AUROC and 0.0006 AUPRC. Schedule settings are therefore inert at
30k, 100k, 300k and 1,114,144: every rung where training succeeds. They matter only at 10k,
where the frozen configuration failed to train at all. The 10k correction is a repair of a
failure rather than a favourable configuration chosen for one rung.

_Run 6 (1,024-token context) was attempted and abandoned. It exhausted the T4's 14.56 GiB
(attention memory is quadratic in sequence length; batch 512 at 1,025 tokens requested 16.03
GiB). Fitting it would require reducing the batch size, which would change two variables at
once and make the 512-to-1024 comparison uninterpretable without also re-running 200 and 512 at
the smaller batch. Section 5.1 therefore reports context length as bounded rather than excluded._
