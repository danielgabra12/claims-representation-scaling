# Stage 5 — Scaling ladder: Arthritis (any subtype)

_Nested patient-level train subsets; fixed temporal test set. Invariants (nesting, size, test-disjointness) asserted for every (size, seed)._

- **Train pool:** 1,114,144 patients. **Fixed test set:** 137,689 (never grows).
- **Seeds per size:** 3.
- **Ladder sizes (local):** [10000, 30000, 100000, 300000, 1000000, 1114144].

## Scale ceiling (honest gate)
- Requested rungs **[3000000]** exceed the local train pool (1,114,144) and are **not reachable on this sample**. The headline >=1M comparison requires the full-population extraction (~10x this sample). This is the crossover study's #1 gate, surfaced here.

Membership: `data/splits/<disease>_ladder.parquet` (seed/rank/PAT_ID); test: `data/splits/<disease>_test.parquet`.