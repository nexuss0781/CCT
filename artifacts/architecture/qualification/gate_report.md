# Architecture Qualification Gate

**Status:** `PASS`
**Checks:** 10/10

- `PASS` **report_complete** — qualification report completed all requested model trials
- `PASS` **real_data_contract** — raw bytes and frozen-tokenizer model-token counts are recorded
- `PASS` **matched_contract** — steps, batch, context, widths, and seed are recorded
- `PASS` **compact_vocab_accounting** — compact vocabulary mode records a smaller active slot allocation
- `PASS` **all_architectures_present** — CCT, GRU, diagonal SSM, and causal attention are all evaluated
- `PASS` **finite_results** — all four model trials have finite metrics
- `PASS` **validation_improvement** — all four models improve held-out validation loss
- `PASS` **test_improvement** — all four models improve frozen test loss
- `PASS` **efficiency_metrics** — parameter, state-memory, and target-token throughput metrics are recorded for every model
- `PASS` **coherent_generation** — every production deterministic no-repeat continuation is full-length and non-repetitive; greedy baselines remain reported
