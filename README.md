# Only one arm's ceiling is estimable

Code and per-run records for *"Only one arm's ceiling is estimable: engineered features versus
learned sequence representations across 10k to 1.1M claims patients."*

Daniel Gabra, Cyril Rakovski. Schmid College of Science and Technology, Chapman University.

## What the paper does

Whether learned representations of clinical event sequences beat hand-engineered features is
usually settled in the few-shot regime. We ask it at scale: 828 engineered features with
gradient-boosted trees (Arm A) against a Deepr-style 1-D CNN (Arm B) and a BEHRT-style
Transformer (Arm C), over identical tokens, on a nested ladder from 10,000 to 1,114,144
training patients, evaluated once on a fixed 2023 holdout of 137,689 patients.

Engineered features lead at every rung. At the top rung the gap to the Transformer is **0.0216
AUROC (95% CI 0.0183 to 0.0247)**. The more useful finding is about curve *shape*: only the
feature arm's asymptote is identifiable across fit windows, and under log-linear gap closure
parity would arrive near 6 to 8 million patients.

The contribution is the evaluation design, not the arthritis number: a pre-specified
information-matching contract, a scale-resolved ladder, identifiability-aware curve fitting that
reports "not identifiable" as a result rather than hiding it, and five ablations each aimed at
one named alternative explanation.

## What is here, and what is not

**Here.** The full pipeline, the extraction SQL with the phenotype definition and eligibility
criteria, the vocabulary construction, per-run configurations and metrics for **all 234 training
runs**, and the analysis scripts that produce every number, table and figure in the paper.

**Not here, and cannot be.** The IQVIA PharMetrics Plus extraction is commercially licensed and
redistribution is prohibited. Researchers with their own licence can reproduce the cohort
exactly from `sql/`. Per-patient test predictions, which back every confidence interval in the
paper, are keyed to identifiers from that licensed extraction and are retained but not released.

This means **the bootstrap intervals cannot be recomputed from this repository alone.** They
need the per-patient predictions. Everything upstream of them can be rerun from the SQL given a
licence, and everything downstream of the committed metrics can be rerun from what is here.

## Layout

```
sql/                 the single extraction: cohort, phenotype, eligibility, vocabulary
src/scale/           the pipeline
  features/          engineered features (828) and the monthly-binned variant (9,672)
  sequences/         tokenisation and the frozen label-free vocabulary
  models/            arm_a (XGBoost), arm_b (CNN), arm_c (Transformer), arm_a_monthly
  scaling/ladder.py  the nested subset ladder
experiments/arthritis/
                     234 runs, each a config.json and a metrics.json
reports/             the analysis scripts
results/             their committed output: every number in the paper
paper/               LaTeX source of the preprint
figures/             Figures 1 and 2
```

## The 234 runs

| group | runs | what it is |
|---|---|---|
| `armA/rich_full`, `armB`, `armC` | 54 | the headline ladder, 6 rungs x 3 seeds x 3 arms |
| `armA/thin_full`, `armA/rich_noutil` | 36 | Arm A feature-set ablations |
| `armA_monthly/*` | 54 | monthly binning, plus 2-bin, 4-bin and the aggregated control |
| `*_r3` | 45 | enrollment-boundary sensitivity (index 2021-2022 only) |
| `*_r2long` | 18 | quadrupled training budget |
| `armC_capL4`, `armC_capD192L4` | 9 | capacity: deeper and wider |
| `armB_len512`, `armC_len512` | 7 | 512-token context |
| `armC_r7cfg` | 6 | schedule-configuration check at 30k and 100k |
| `armC_r4fix` | 5 | corrected 10k rung, 5 seeds |
| **total** | **234** | |

Every run directory holds the exact configuration it ran under, so any cell in any table traces
to a file.

## Reproducing the analysis

Given the committed run records, every figure and table regenerates without the licensed data:

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python reports/scaling_analysis.py    # learning curves, power-law fits, identifiability
python reports/ablations.py           # context, capacity, schedule
python reports/armAM_analysis.py      # monthly temporal resolution
python reports/run5_binning.py        # the 2/4/12-bin sweep
python reports/figures_paper.py       # Figures 1 and 2
```

`reports/paired_bootstrap.py` and `reports/operating_points.py` need the per-patient predictions
and will not run from this repository; their committed output is in `results/`.

## Rebuilding the cohort from scratch

You need your own IQVIA PharMetrics Plus licence.

1. Run `sql/01_extract.sql` (templated; `sql/render.py` fills in your table names) against your
   instance. The phenotype is ICD-10 M05-M08, M10, M12, M13, M15-M19, M45, L40.5, requiring two
   distinct calendar months with a qualifying diagnosis, excluding anyone prevalent at index.
2. `python -m src.scale.scaling.ladder` builds the nested subsets.
3. `python -m src.scale.features.engineered` and `python -m src.scale.sequences.build` build the
   two representations under the matching contract in `results/fairness_contract.md`.
4. `python -m src.scale.models.arm_a` (and `arm_b`, `arm_c`) train the arms.

Arm A trains on CPU. Arms B and C want a GPU; everything in the paper ran on a single NVIDIA T4
in roughly 40 GPU-hours total.

## Citation

See `CITATION.cff`. The preprint is on arXiv; this README will carry the identifier once it is
assigned.

## Licence

Code is MIT (see `LICENSE`). It does not cover the IQVIA data, which is not redistributable.
