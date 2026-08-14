# Level 1 English Acquisition Milestone

**Position:** First training milestone after the L1-8 operation contract, before bounded teaching behavior.
**Implementation:** Native C++20 only.
**Status:** Active implementation contract.

## Objective

Train the CCT language core on real English text until it demonstrates measurable, reproducible English-language modeling and grammatical preference on held-out data. In this milestone, “native English speaker” is operationalized as a measurable English acquisition target: lower held-out language-model loss than an untouched control, stable generation primitives, and above-chance preference for grammatical sentences. The phrase is not treated as a hidden or subjective pass criterion.

## Fixed data contract

| Role | Dataset | Use | Isolation |
|---|---|---|---|
| Pretraining | Pinned WikiText-2 raw source from the Track 1 native preparation, capped at 2,000,000 train tokens | Native next-token training only | Train text is never used for final scoring. |
| Selection validation | Pinned WikiText-2 validation split | Optimizer-step selection and early diagnostics | Never used for parameter updates. |
| Frozen language test | Pinned WikiText-2 test split | Final next-token loss/perplexity and token accuracy | Scored once after the release candidate is frozen. |
| Grammar evaluation | BLiMP, 67 minimal-pair JSONL files, pinned repository archive hash | Native causal scoring of acceptable versus unacceptable English | Evaluation-only; no BLiMP sentence is used for training. |
| Downstream readiness | Existing Track 1 SQuAD/SFT bundle | Held for the later instruction stage | Not mixed into the language-acquisition test. |

The first bounded run may use a declared token and pair cap for feasibility, but the cap, source hash, file count, and selection rule must be written into the artifact manifest. A capped run is an acquisition experiment, not a claim of full-corpus competence.

## Required implementation

The native runner must load the frozen tokenizer, build identity-checked causal datasets, train from a deterministic no-training control and fixed seed, publish an atomic pretraining checkpoint, evaluate selection and frozen language-test slices, and score BLiMP pairs through the public `next_logits`/loss interface. No Python training or evaluation code is permitted.

The evaluator must record aggregate BLiMP accuracy, per-file accuracy, the four declared BLiMP fields where present, pair counts, truncations, non-finite counts, and exact source/tokenizer/checkpoint identities. Every sentence must be normalized through the frozen native tokenizer and every score must use the same causal token-loss convention.

## Gate requirements

A candidate English-acquisition run is `PASS` only if all mandatory conditions hold:

1. source, tokenizer, dataset, configuration, and checkpoint identities are complete and internally consistent;
2. all fixed-split and evaluator-only boundaries are zero-overlap;
3. training, validation, test, and BLiMP metrics are finite;
4. trained held-out WikiText loss improves over the no-training control on every declared seed;
5. the frozen language test is scored after training and is not used for updates or checkpoint selection;
6. BLiMP accuracy is above the 50% chance baseline and improves over the matched no-training control, with no declared field below the configured floor;
7. no malformed sentence, unsupported token, oversized context, or corrupted checkpoint is silently accepted;
8. repeated same-seed execution reproduces checkpoint and report identity; and
9. all native unit, gate, strict-warning, and repository regression tests remain green.

Initial thresholds are configuration values rather than unvalidated claims of human equivalence. The release report must state the exact corpus cap, training steps, model dimensions, seed set, BLiMP pair cap, metrics, controls, and known limitations.

## Failure paths

The runner must fail closed on missing source cache, source hash mismatch, tokenizer mismatch, empty split, train/test overlap, evaluator data in training, non-finite metrics, invalid token IDs, context overflow, zero active targets, stale checkpoints, trailing checkpoint data, and any attempt to use frozen test or BLiMP data for updates.

## Artifacts and transition

The release bundle must contain the frozen configuration, source manifest, dataset identities, training report, checkpoint hash, selection/test metrics, BLiMP aggregate and per-file reports, control comparison, failure report, and a human-readable report. `PASS` authorizes a later instruction/teaching milestone only after explicit approval; it does not establish broad language competence, human preference equivalence, factuality, production readiness, or general intelligence.

## External references

The WikiText dataset card describes the English WikiText raw corpus and its fixed language-modeling splits; this executed release uses the existing pinned WikiText-2 Track 1 source rather than claiming a WikiText-103 acquisition.[1] BLiMP defines 67 English minimal-pair datasets with 1,000 pairs per file and evaluates whether a language model assigns higher probability to the acceptable sentence.[2] The BLiMP paper provides the benchmark methodology and human-validation context.[3]

## References

[1]: https://huggingface.co/datasets/Salesforce/wikitext "Salesforce WikiText dataset card"

[2]: https://github.com/alexwarstadt/blimp "BLiMP benchmark repository"

[3]: https://aclanthology.org/2020.tacl-1.25/ "BLiMP: The Benchmark of Linguistic Minimal Pairs for English"
