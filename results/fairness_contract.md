# Fairness contract — Arm A vs Arm B (scale-crossover study)

_The written spec both model arms are built against. The crossover paper's headline is
"at what training size (if any) do learned sequence representations overtake engineered
features + GBDT." For that comparison to be honest, the two arms must differ in
**representation**, not in the **information** they are allowed to see. This document
freezes what each arm gets and how each is tuned, so neither can be accused of being a
straw man._

> **UPDATED 2026-08-17.** This document is the pre-registered protocol; the sections below
> are kept as agreed, with corrections marked. Three things in the original text are now
> stale and are corrected inline: the test set size, the claim that a Transformer was
> computationally infeasible (Arm C exists and is the paper's third arm), and the scope
> (arthritis only; t2d is cut). Section 9 records the asymmetries we later measured and
> failed to anticipate here, which is the honest part of a fairness contract.

Agreed 2026-07-12 (Daniel). Treatment-exclusion choices below align with the phenotype/
treatment definition that is still **PENDING Cyril sign-off**; they are documented and
overridable, not silently baked in.

## 1. Information both arms see (identical)

Both arms read the **same source**: every claim in the per-patient baseline window
`[FIRST_MONTH, index)` (12 months, strictly pre-index), for the same cohort, same
temporal holdout (train index-years 2020–22, test 2023), same fixed test set, same
nested scaling ladder, same 3 seeds.

- **Arm A** compresses that window into order-free engineered features (this document,
  Stage 3.1).
- **Arm B** consumes the same claims as an **ordered sequence** and learns its own
  representation (Stage 7).

The only licensed difference is order/representation. If Arm B wins, it must be because
sequence structure carries signal that aggregation discards — not because Arm A was
handed a thinner slice of the data.

## 2. Arm A feature families (Stage 3.1)

| family | # (approx) | content | in access-confound ablation? |
|---|---|---|---|
| demo | 7 | age, sex, region | kept |
| comorbidity | 12 | curated ICD-10 presence flags (Elixhauser-inspired + MSK symptoms) | kept |
| dx_bag | ≤500 | count per 3-char ICD-10 category, top-K by **train-pool patient frequency** | kept (clinical content) |
| rx_class | ~100 | count per therapeutic class (`rx_lookup.THPTC_CLAS_DESC`) | kept (clinical content) |
| proc_bag | ≤200 | count per procedure code, top-K by train-pool patient frequency | kept (clinical content) |
| utilization | 8 | pure volume/intensity counts (`n_claims`, `n_dx`, …) | **removed** |
| temporal | 5 | recency/trend (months since last claim, recent-quarter vs prior volume) | **removed** |

`thin` = demo + comorbidity + utilization (the original 27-feature Arm A) is retained as
a **regression baseline** so we can measure how much enrichment bought.

## 3. Leakage rules (audited every build)

- Every feature is computed only from claims strictly before index (existing assertion,
  extended to the new blocks).
- **Vocabulary is chosen label-free and test-free:** top-K by raw patient frequency in
  the **train pool only**, frozen to `*_features_vocab.json`, then applied identically to
  every ladder size, seed, and the test set. Predictive selection (which uses labels) is
  left to the trees.
- **Phenotype codes cannot enter the bag.** The cohort is disease-free at index, so any
  qualifying dx in baseline would have excluded the patient as prevalent. A defensive
  assertion drops any vocab category matching the disease dx-regex regardless.

## 4. Treatment exclusion (decision #1 — keep symptom signal, drop disease-specific drugs)

A pre-index NSAID/analgesic is legitimate pre-symptomatic signal and is **kept**. A
pre-index disease-specific DMARD/biologic/urate-lowering agent effectively encodes
already-present (uncoded) disease and is **removed** from the rx_class block. For
arthritis this is `rx_lookup.USC_NAME ∈ {ANTIARTH, DMARDS; ANTIARTH, BIOLOGICAL RESPONSE
MODIFIERS; ANTIARTH, GOUT SPECIFIC}`. Multi-indication drugs (methotrexate,
hydroxychloroquine, sulfasalazine) are **kept** — excluding them would also delete their
oncology/malaria/IBD signal, and they are not arthritis-specific. Per-disease exclusion
lists live in `config.RX_TREATMENT_EXCLUDE_USC` and are subject to Cyril sign-off.

Procedures are treated as a plain bag-of-codes (no treatment exclusion in v1);
procedure-level treatment exclusions (e.g. intra-articular injection) are a noted
candidate refinement, deferred.

## 5. Temporal signal budget (decision #2 — keep it light)

Arm A gets only the 5 light recency features above. Richer temporal structure (ordering,
inter-event gaps, sequence position) is **reserved for Arm B by design** — handing it to
Arm A would blur the very contrast the study measures.

## 6. Tuning budget (symmetric, documented)

Parity rule: **neither arm gets a label-tuned advantage over the other.** As built, both
arms run a **single fixed, documented configuration** — no per-size or per-seed search on
either side — with early stopping on a train-carved validation split (10%, the same carve
for both arms). Because both are fixed rather than searched, the comparison is symmetric.

A modest, *equal* validation search (a handful of configs per arm, best frozen across all
sizes/seeds) is a documented future refinement; if done, it must be done for **both** arms
under the same budget. The contract is "each arm is tuned to the same degree," which today
means "both fixed," not "one searched and the other fixed."

## 7. Arm B model and compute budget (Stage 7)

Arm B reads the ordered code sequence over the same baseline window, tokenized with **Arm
A's exact frozen vocabulary** (same 500 DX cats + 200 procedures + 95 drug classes, same
treatment exclusion; out-of-vocab → UNK), plus the **same 7 demographics**. Sequences keep
real month order and are truncated to the most-recent 200 tokens.

Model: `Embedding + learned positional encoding → multi-kernel 1D CNN (Deepr-style; Nguyen
et al. 2017, kernels {3,5}, 64 filters) → masked global max-pool → concat demographics →
MLP → logit`. Fixed budget: Adam, lr 1e-3, batch 512, ≤25 epochs, early-stop patience 4 on
val AUROC, no positive-class reweighting (matching Arm A).

**Architecture history (honest, CPU-only box — no GPU, ~8 cores).** A BiGRU and a small
Transformer were both implemented and measured first; each cost **~13–15 h per run** on
this hardware (RNNs are sequential over timesteps; the Transformer's padding-mask path
falls off PyTorch's CPU fast path), which is infeasible for a 15-run ladder × 3 seeds. The
1D CNN is pure batched convolution (~5 min/epoch at the top rung) and still reads order
via local n-gram filters + positional embeddings. This is a **compute** choice, not a
representational compromise: Arm B remains an order-aware learned representation contrasted
with Arm A's order-free bag. If the full-population extraction and a GPU become available,
re-running Arm B as a Transformer is a documented, drop-in upgrade.

> **CORRECTED.** This happened. A GPU (Colab T4) became available and **Arm C**, a
> BEHRT-style Transformer encoder over the identical tokens, vocabulary, ladder, seeds,
> validation carve and test set, was run to completion (18 runs, 11.8 GPU-hours). The
> "computationally infeasible" statement above describes the CPU-only situation in July
> 2026 and no longer holds. Arm C's configuration: d_model 128, 8 heads, 2 layers, ff 256,
> dropout 0.2, pre-norm, CLS pooling, AdamW lr 3e-4, wd 1e-2, 300-step warmup, grad-clip
> 1.0, batch 512, <=25 epochs, patience 4. **410,497 parameters.**

## 8. Evaluation (identical, frozen)

Temporal holdout, fixed test set, 3 seeds, metrics AUROC + AUPRC + Brier.

> **CORRECTED.** The 61,011 figure was the 10%-sample study. At the scale extraction the
> fixed test set is **137,689 patients (2,554 incident, 1.85%)** drawn from index year 2023,
> and the train pool is 1,114,144. The thin-baseline regression guard referenced below was
> also a 10%-sample number; at scale, thin reaches 0.8226 and rich_full 0.8279.
AUPRC is the honest headline for the ~2% base rate. **Regression guard:** enriched Arm A
must meet or beat the thin baseline (at scale: AUROC 0.8226). If enrichment does not help, that is
a real finding (engineered features have hit their ceiling), reported as such.

---

## 9. Where the contract was violated in practice (added 2026-08-17)

A fairness contract is only worth anything if the failures are reported. Three asymmetries
survived the design and were found afterwards by audit. All are quantified in the paper.

1. **Context truncation, favouring Arm A.** Sequences are capped at 200 tokens while Arm A
   aggregates the window uncapped. On the test set this truncates **40.1% of incident cases**
   against **17.2% of non-cases**, so the sequence arms saw a censored view of the class that
   determines AUPRC. Measured at 512 tokens: the Transformer gains +0.0045 AUROC at 300k
   (95% CI +0.0017 to +0.0072) and +0.0032 at the top rung, closing 15% of the AUROC gap and
   35% of the AUPRC gap. Real, and not sufficient to explain the result.
2. **Out-of-vocabulary handling, favouring Arms B and C.** Codes outside the frozen vocabulary
   are dropped entirely by Arm A but keep a positional UNK placeholder in the sequence arms.
   This affects 7.2% of diagnosis rows and 35.8% of procedure rows. Not separately ablated.
3. **Excluded treatments leak back into the sequence arms.** DMARDs, biologics and
   urate-lowering agents are removed from Arm A's rx block entirely, but in the sequence
   arms the claim still occupies a position as UNK. The signal is diluted among all other
   out-of-vocabulary tokens, but the exclusion is not symmetric across arms.

**Tuning parity is also weaker than section 6 claims.** Both arms run one fixed configuration,
which is symmetric in form but not in effect: gradient-boosted trees are far more robust to
being untuned than transformers are. An equal-budget validation search on both sides would be
a stronger guarantee than "both fixed".

**Arm A-M (added 2026-08-17).** Because claims are month granular and within-month order is
arbitrary, the only information the sequence arms hold that Arm A lacks is which of the 12
observation months an event fell in. Arm A-M gives the bag exactly that (9,672 columns, all
count features split into 12 monthly bins, same vocabulary, same exclusions, same untuned
XGBoost configuration) and is the information-equivalence control this contract should have
specified from the start.
