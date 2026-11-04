# Stage 3.2 — Code sequences (Arm B inputs): Arthritis (any subtype)

_Same frozen vocabulary as Arm A (identical information); tokens kept in real month order. Baseline window only; vocab frozen from the train pool._

- **Patients:** 1,251,833. **Vocab size:** 798 (incl. PAD, UNK). **Max length:** 1024 (most-recent tokens kept).
- **Sequence length:** median 51, 95th pct 490.
- **No-baseline-claim patients:** 196,734 (single PAD token; demographics only).

Table: `data/processed/arthritis_sequences_len1024.parquet` (gitignored). Token map: `arthritis_sequences_len1024_vocab.json`.