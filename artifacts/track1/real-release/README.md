# Track 1 real-source replay

This bundle records a fresh native C++20 Track 1 preparation and bounded training replay from the pinned public sources declared in `manifest.json`. The native source lineage is commit `baa3bc7` (`fix(l1-6): support multi-answer real-source ingestion`).

The preparation used WikiText-2 raw-v1 and GEM SQuAD-v2 source revisions recorded in the manifest, with selection seed `1701`, a `200000` byte-fallback-token pretraining cap, `900` SFT training examples, `100` SFT evaluation examples, and `11873` frozen final-test examples. The native trainer then ran `40` pretraining steps and `40` SFT steps with context length `32`, embedding dimension `4`, hidden dimension `4`, target-span-only SFT masking, and final-test limit `2000`.

The durable identity chain is `manifest.json` -> `training_report.json` -> `real-release-training/pretrain_checkpoint.bin` -> `real-release-training/sft_checkpoint.bin`. The training report records the manifest, pretraining dataset, SFT dataset, final-test, tokenizer, and checkpoint hashes, plus `training_authorized: false` and `checkpoint_lineage: pretrain_checkpoint->sft_checkpoint`.

Prepared data and checkpoint artifacts are included for native replay. Oversized raw source payloads are not duplicated into the GitHub release bundle; their pinned URLs, revisions, licenses, raw digests, and attestation digests remain recorded in `manifest.json`, while `.sha256` sidecars preserve the acquired-payload hashes. This is an artifact-size control, not a change to source identity or split policy.

The reported metrics are bounded answer-target next-token metrics. This replay does not claim exact-match/F1 answer quality, broad language competence, production readiness, or general intelligence.
