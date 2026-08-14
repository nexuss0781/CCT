# Stage 11 Expanded Gate Contract

## Trainable Native NLP Core

**Predecessor:** Stage 10 — Tokenizer and Representation Engine
**Successor:** Stage 12 — Scaling and Accelerator Systems
**Implementation boundary:** Native C++20 only; no Python trainer, optimizer, dataset reader, or gate is accepted.
**Gate status:** Native C++20 implementation complete; the current formal gate is PASS with thirteen mandatory checks and explicit approval still required for Stage 12.

## 1. Gate purpose

Stage 11 tests whether the CCT selective recurrent state can optimize a real categorical next-token objective over governed token streams while preserving causal masks, tokenizer identity, deterministic checkpoints, matched controls, and fail-closed numerical behavior. The gate is deliberately a small controlled pilot. It does not claim broad language competence, scale efficiency, factuality, safety, or production usefulness.

A passing gate authorizes Stage 12 scaling and accelerator work only. It does not authorize instruction tuning, preference optimization, retrieval, deployment, or unrestricted training.

## 2. Frozen data and provenance contract

The gate must run the Stage 10 tokenizer from its immutable snapshot and require the exact Stage 10 snapshot hash before constructing any training example. The input corpus is rebuilt from the Stage 9 governed manifest and includes real Project Gutenberg text fixtures, real CCT C++ fixtures, code/JSON/Unicode/separator application fixtures, and an evaluator-only canary. Vocabulary and training streams may consume only train-split records with `training_allowed=true`; validation and evaluator-only records must be excluded from training and checkpointed dataset hashes.

| Fixture family | Use | Required evidence |
|---|---|---|
| Project Gutenberg `pg1342.txt` | Train | Real-source hash and source record identity retained |
| Project Gutenberg `pg11.txt` | Validation only | Never affects training parameters or vocabulary |
| CCT `production.cpp` and `corpus.cpp` | Train | Code identifiers, comments, punctuation, and long-context bytes |
| Application code/JSON fixture | Train | Identifier, indentation, literal, delimiter, and escape behavior |
| Unicode/malformed separator fixture | Train regression | Valid Unicode, malformed UTF-8, NUL, tabs, CRLF, and repeated boundaries |
| Evaluator-only canary | Evaluation/contamination only | Builder rejects it; no token or parameter mutation |

The dataset reader must expose record ID, shard ID, cursor, tokenizer version/hash, source offsets, target IDs, and loss masks. A checkpoint must record the dataset fingerprint and cursor. Any changed source hash, tokenizer hash, split assignment, or evaluator inclusion fails closed.

## 3. Model contract

The CCT candidate must use a selective recurrent state as its temporal backbone. The released Stage 11 implementation uses trainable token embeddings, selective retain/write gates, a nonlinear candidate state, an output projection, stable softmax cross-entropy, analytic backpropagation through time, gradient clipping, and AdamW-equivalent updates. The configuration records vocabulary size, embedding/hidden dimensions, candidate kind, seed, context length, optimizer, schedule, and tokenizer hash.

The gate compares CCT against three controls on the same token IDs, context limit, training token budget, optimizer step budget, seeds, and hardware class: dense causal attention, GRU, and diagonal SSM. Controls must complete without numerical failure and must report parameter counts, training tokens, wall time, throughput, and memory/state accounting. The gate does not permit a control to be omitted because its result is inconvenient.

## 4. Training protocol

The fixed pilot uses three seeds, a small deterministic context limit, a fixed number of optimizer steps, a fixed batch/document order, and a declared AdamW-equivalent schedule with warmup, linear decay, weight decay, and global gradient clipping. Every accepted run records initial and final training loss, validation loss, perplexity, token count, gradient norms, learning-rate schedule, parameter count, elapsed time, tokens/sec, state-memory bytes, and checkpoint hashes.

The accepted CCT run must reduce validation loss for **all three seeds** relative to the corresponding untrained initialization. The mean validation loss must improve by at least **5%** from initialization and the selected CCT result must beat the no-training control by at least **1%**. These thresholds are measured on the fixed held-out validation documents and are not language-model quality thresholds.

A tiny repeated training fixture must overfit sufficiently to demonstrate that the objective and optimizer can learn: final training loss must be at least **25% lower** than its initial loss while remaining finite. The gate stops and records `FAIL` on any NaN, Inf, unexplained loss increase beyond the declared diagnostic tolerance, or gradient norm overflow.

## 5. Gradient and objective contract

The objective is mean categorical cross-entropy over positions with `loss_mask=true`. The final position of every document and every padding position is ignored. The implementation must use a numerically stable log-sum-exp calculation and report finite perplexity `exp(loss)` only when the loss is finite and bounded.

The analytic gradient is checked against centered finite differences on selected embedding, recurrent, gate, and output parameters using a small hand-computable fixture. The maximum relative error threshold is **1e-4** for non-negligible gradients, with an absolute tolerance of **1e-6** near zero. Gradient clipping must be deterministic and the optimizer update direction must reduce the checked loss for a small step.

## 6. Checkpoint and resume contract

Checkpoint V2+ must contain model configuration, model kind, tokenizer version/hash, dataset hash, train/validation split identity, parameter vector, optimizer moments, optimizer step, scheduler state, data cursor, RNG state/seed, metrics, and a canonical serialized hash. Truncated, malformed, incompatible, wrong-tokenizer, wrong-dataset, duplicate-field, and non-finite checkpoints must reject before model use.

The harness interrupts training at cursors `0`, `1`, and a mid-epoch position, saves a checkpoint, reloads it, and resumes with the same remaining documents. The resumed and uninterrupted runs must have identical cursor, optimizer step, parameter vector, optimizer moments, and final validation metrics within **1e-12** for deterministic CPU execution. No record may be duplicated or skipped outside the explicitly declared end-of-epoch policy.

## 7. Mandatory checks and hard thresholds

| Check | Hard pass condition |
|---|---|
| Objective | Stable finite cross-entropy, token count, and perplexity with exact loss masks |
| Optimization | All three CCT seeds improve held-out validation loss; mean improvement ≥5% |
| Gradient | Analytic/finite-difference maximum relative error ≤1e-4, with absolute tolerance ≤1e-6 near zero |
| Stability | No NaN/Inf, non-finite logits, non-binary masks, unsupported model kinds, or accepted divergent run |
| Checkpoint | Canonical checkpoint hash, complete fields, malformed/truncated/corrupt/wrong-identity rejection |
| Resume | Cursor-0, cursor-1, and mid-epoch interruption agree with uninterrupted training within 1e-12 |
| Data cursor | No duplicate or skipped record beyond declared final cursor policy; model/dataset context and optimizer budget are enforced |
| Baselines | Dense attention, GRU, and diagonal SSM all complete under identical declared budgets |
| Capability | Selected CCT beats its no-training control by ≥1% validation-loss improvement |
| Regression | Complete prior Stage 0–10 CI remains green |
| Efficiency | Every model reports positive tokens/sec, elapsed time, peak/resident memory, and state size |
| Reproducibility | Three seeds and repeated same-seed runs reproduce within declared deterministic tolerance |
| Artifact integrity | Tokenizer, corpus, config, checkpoint, dataset, and metric hashes are present and consistent |

Additional hard limits are fixed for the gate environment: the selected CCT run must process at least **100 tokens/sec**, use a positive measured state-memory value, and keep the accepted parameter count within **±10%** of the matched control band. If the real-source pilot cannot satisfy a threshold without changing the frozen data or budget, the gate returns `BLOCKED` or `FAIL`; it must not silently relax the contract.

## 8. Adversarial and failure-path requirements

The harness must attempt evaluator-only dataset construction, duplicate record identity, changed tokenizer hash, changed corpus hash, invalid target ID, non-binary/all-false loss masks, malformed/truncated/corrupt checkpoint, trailing checkpoint data, incompatible model kind, non-finite parameter/gradient/optimizer injection, over-capacity model allocation, dataset/model context mismatch, optimizer-budget overrun, wrong batch shape, cursor regression, and cross-document target leakage. Each must reject deterministically or record a fail-closed diagnostic.

The harness must verify that changing validation or evaluator content after model construction cannot mutate train parameters, dataset hash, tokenizer hash, or checkpoint identity. The baseline accounting must verify model names, parameter counts, context length, optimizer step budget, token budget, and seed metadata rather than comparing only final loss.

## 9. Required machine-readable artifacts

The gate writes all artifacts beneath `artifacts/stage-11/cpp-gate/`:

| Artifact | Required contents |
|---|---|
| `checks.json` | One record per mandatory check with status, duration, and measured evidence |
| `metrics.json` | Mandatory-check count, objective, loss, perplexity, gradient, stability, throughput, memory, and parameter metrics |
| `seed_comparison.json` | Three-seed CCT initialization/final validation table |
| `baseline_comparison.json` | Dense attention, GRU, diagonal SSM, and CCT matched accounting |
| `checkpoint_report.json` | Checkpoint fields, hashes, interruption cursors, and resume equivalence |
| `dataset_manifest.json` | Real-source hashes, splits, token counts, shard IDs, tokenizer hash, and evaluator exclusion |
| `gradient_report.json` | Hand fixture and finite-difference parameter checks |
| `resource_profile.json` | Tokens/sec, wall time, state memory, parameter memory, and hardware class |
| `incident_log.json` | Numerical, cursor, contamination, version, and boundary incidents |
| `release_record.json` | Stage status, selected model/checkpoint hashes, training authorization boundary, and approval requirement |
| `report.md` | Human-readable evidence, baseline comparison, limitations, and claim boundary |

## 10. Transition decision

The gate returns `PASS` only when every mandatory check passes, a real CCT next-token run is stable and improves held-out loss, all three matched controls complete, interrupted/resumed training is equivalent, artifact hashes are complete, and the report states that the evidence is a small controlled pilot. A `FAIL` requires remediation. A `BLOCKED` result must name the unresolved hardware or data-rights constraint and must not authorize Stage 12.

Stage 11 does not prove broad language competence, scale efficiency, factuality, safety, instruction following, retrieval grounding, production usefulness, or general intelligence. Those claims remain outside scope and require later independent gates.
