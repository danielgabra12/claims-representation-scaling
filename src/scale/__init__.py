"""scale — the data-scale crossover study.

A clean, config-driven pipeline (spec: scale_crossover_research_spec.md) that measures
at what training-set size, if any, learned sequence representations (Arm B, PyTorch)
overtake engineered-features + gradient-boosted trees (Arm A) for clinical onset
prediction, under a temporal holdout and across diseases.

This package is deliberately separate from the older ``src/`` epi code and the
superseded ``src/ml/`` 4-tier tree. Reusable logic (ICD groupings, leakage
assertions) is ported here explicitly rather than imported, so provenance is clear.

Module map (mirrors the spec's Stage layout):
    config      — single source of truth: seeds, paths, ladder, study design, raw schema
    data/       — DuckDB out-of-core access + data manifest (Stage 0/1)
    features/   — engineered per-patient features, Arm A inputs (Stage 3)
    sequences/  — ordered code histories, Arm B inputs (Stage 4)
    models/     — Arm A (GBDT) and Arm B (sequence encoder) (Stages 6/7)
    scaling/    — nested patient-level subset ladder (Stage 5)
    evaluation/ — metrics, calibration, crossover analysis (Stage 8)
"""
