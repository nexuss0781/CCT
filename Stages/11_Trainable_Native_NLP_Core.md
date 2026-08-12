# Stage 11 — Trainable Native NLP Core
## Real Next-Token Modeling, Checkpoints, and Matched Controls

**Predecessor:** Stage 10 — Tokenizer and Representation Engine  
**Successor:** Stage 12 — Scaling and Accelerator Systems  
**Status:** Implemented and gated

**Implementation:** Native C++20 trainer, categorical objective, analytic CCT BPTT, AdamW-equivalent optimizer, checkpoints, tests, and gate

## Purpose

Stage 11 establishes a real native next-token training contract around the immutable Stage 10 tokenizer snapshot. It tests whether the CCT selective recurrent state can optimize a categorical language objective on a governed, bounded pilot while preserving causal masks, provenance, deterministic optimizer state, checkpoint recovery, matched controls, and fail-closed numerical behavior.

This is a controlled CPU pilot and not a claim of broad language competence, production language-model quality, scale efficiency, factuality, safety, or general intelligence.

## Scope and non-goals

The stage covers governed token-window construction, sparse-domain embeddings, causal next-token cross-entropy, analytic CCT recurrence gradients, optimizer state, scheduling, anomaly rejection, validation, matched controls, checkpoint recovery, and native artifact production. It does not cover large-scale distributed training, accelerator kernels, instruction tuning, preference alignment, retrieval, serving, deployment, human evaluation, or unrestricted training.

## Implemented scope

The implementation is in `cpp/include/cct/nlp_trainer.hpp` and `cpp/src/nlp_trainer.cpp`. It provides explicit train/evaluation eligibility flags on encoded documents; deterministic context windows with final-position and boundary loss masking; sparse token-ID-domain handling for the Stage 10 snapshot; trainable token embeddings; CCT selective retain/write recurrence; stable log-sum-exp softmax cross-entropy; analytic backpropagation through the CCT recurrence; global gradient clipping; warmup and linear decay; AdamW-equivalent first/second moments; validation loss, perplexity, accuracy, throughput, and state-memory metrics; matched dense causal attention, GRU, and diagonal SSM controls; and canonical checkpoint V2 serialization with tokenizer, dataset, optimizer, scheduler, cursor, RNG-state, history, model, and optimizer-moment fields.

Evaluator-only records, records without explicit training permission, non-finite model parameters, invalid token IDs, all-false loss masks, malformed checkpoints, incompatible tokenizer/dataset identities, truncated model state, and optimizer-state size mismatches reject closed rather than being coerced into training.

## Frozen pilot contract

The gate binds the trainer to the tracked Stage 10 snapshot at `data/stage-10/tokenizer_snapshot.bin`. The snapshot hash is `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a`. The released snapshot is the hybrid candidate with a sparse token-ID domain of 768 IDs and 521 serialized vocabulary rows.

The pilot uses real Project Gutenberg training/validation fixtures, real native C++ source fixtures, application-shaped code/JSON/Unicode/separator fixtures, and an evaluator-only canary. Training documents have explicit `training_allowed=true`; validation documents have explicit `evaluation_allowed=true`; evaluator-only records are rejected from both training and validation dataset construction. The resulting pilot has 37 training windows, 18 validation windows, 782 active training targets, and 389 active validation targets. Its dataset identity is `d8c7c24937e7603064a1a5d3b07b0472fe672426b52d00e41b2b4d614c240996`.

The fixed CCT pilot uses embedding dimension 2, hidden dimension 2, context length 24, 120 optimizer steps, learning rate 0.04, two warmup steps, linear decay, clip norm 2.0, no weight decay, and three seeds `3`, `5`, and `7`. The dimensions are chosen to keep the selected CCT parameter count within the matched-control band rather than granting it a larger model.

## Model and training contracts

```text
forward(token_ids, causal_boundaries) -> logits, recurrent state
loss(logits, targets, loss_mask) -> mean categorical cross-entropy, token count
backward(loss) -> analytic parameter gradients
optimizer_step(gradients) -> updated parameters and moments
checkpoint() -> model, optimizer, scheduler, RNG, cursor, config, tokenizer hash, dataset hash
resume(checkpoint) -> deterministic equivalent training state
```

The CCT candidate uses trainable embeddings, selective retain/write gates, a nonlinear candidate state with a previous-input projection, and an untied vocabulary output head. The matched controls use the same sparse token-ID domain, context length, pilot data, and declared optimizer budget. Their dimensions and parameter counts are recorded in the gate artifacts.

| Model | Parameters | State memory | Final pilot evidence |
|---|---:|---:|---|
| CCT | 3,862 | 16 bytes | Three-seed validation improvement; selected seed improvement 13.889% |
| Dense causal attention | 3,852 | 784 bytes | Completed matched evaluation and training budget |
| GRU | 3,870 | 16 bytes | Completed matched evaluation and training budget |
| Diagonal SSM | 3,846 | 16 bytes | Completed matched evaluation and training budget |

## Mandatory gate checks

The Stage 11 gate contains eight application-shaped mandatory checks. All checks passed.

| Check | Result | Evidence |
|---|---|---|
| Tokenizer and dataset identity | **PASS** | Exact Stage 10 snapshot hash and governed dataset hash |
| Objective, analytic gradient, optimizer | **PASS** | Finite categorical cross-entropy; finite-difference relative error at or below `1e-4` |
| Three-seed CCT validation pilot | **PASS** | Seed improvements: 13.889%, 15.167%, and 14.141% |
| Repeated-corpus overfit | **PASS** | 70.682% training-loss reduction |
| Matched controls | **PASS** | Dense attention, GRU, and diagonal SSM all completed |
| Checkpoint interruption/resume | **PASS** | Cursors 0, 1, and 3 agree with uninterrupted training within `1e-12` |
| Contamination and invalid-input controls | **PASS** | Evaluator-only, invalid-target, and all-false-mask rejection |
| Artifact identity and checkpoint integrity | **PASS** | Checkpoint hash `8ff1f227513d79a840b648bd724823e3fd790ba3bd9e754a086f430ebbd81b62` |

The selected seed-3 run improved validation cross-entropy from `6.877456` to `5.922243`, a relative improvement of `13.889%`, with final validation perplexity `373.248` and measured validation throughput of approximately `80,641` tokens/sec in the declared CPU environment. These are bounded pilot measurements, not production performance claims.

## Evaluation harness

The native regression suite checks objective masking, finite metrics, analytic/finite-difference gradients, optimizer schedule, deterministic initialization, matched controls, checkpoint resume, wrong-identity rejection, malformed checkpoints, all-false masks, and non-finite parameters. The artifact-producing gate adds the real-source/application-shaped pilot, three seeds, held-out validation, repeated-corpus overfit, baseline accounting, multiple interruption cursors, evaluator-only rejection, parameter-band enforcement, and machine-readable release records.

## Regression and CI integration

The native regression executable is `cct_nlp_trainer_tests` and covers objective masking, finite metrics, analytic/finite-difference gradients, optimizer schedule, deterministic initialization, all matched controls, checkpoint resume at exact state, wrong-identity rejection, malformed checkpoints, all-false masks, and non-finite parameters. The Stage 11 gate executable is `cct_stage11_gate`.

The canonical commands are:

```bash
make stage11-test
make stage11-gate
make ci-stage11
```

`ci-stage11` runs the complete sequential Stage 0–10 chain, then the Stage 11 regression suite and gate. Its final release run must be performed from the immutable release commit after documentation and compatibility changes are complete.

## Deliverables

The gate writes machine-readable and human-readable evidence under `artifacts/stage-11/cpp-gate/`:

| Artifact | Purpose |
|---|---|
| `checks.json` | Mandatory check status and measured evidence |
| `metrics.json` | Loss, improvement, parameter-band, and status metrics |
| `seed_comparison.json` | Three-seed CCT results |
| `baseline_comparison.json` | Dense attention, GRU, diagonal SSM accounting |
| `checkpoint_report.json` | Hash, identities, and interruption/resume evidence |
| `dataset_manifest.json` | Tokenizer/dataset identity, split counts, and evaluator exclusion |
| `gradient_report.json` | Objective and gradient tolerance declaration |
| `resource_profile.json` | Parameter, state-memory, and throughput metrics |
| `incident_log.json` | Numerical, identity, cursor, contamination, and boundary incidents |
| `release_record.json` | Stage status and transition boundary |
| `selected_checkpoint.bin` | Selected native trainer checkpoint |
| `report.md` | Human-readable evidence and claim boundary |

## Pass/fail transition

Stage 11 passes because the real next-token CCT pilot is finite and trainable, all three seeds improve held-out validation loss by more than the declared 5% threshold, the repeated corpus overfit test passes, analytic gradients agree with finite differences, all matched controls complete, checkpoints resume exactly at multiple cursors, evaluator-only data is rejected, and all artifacts are internally identified.

A Stage 11 `PASS` authorizes Stage 12 scaling and accelerator work within its own specification. It does not authorize instruction tuning, preference optimization, retrieval, production serving, unrestricted training, or deployment. Training authorization remains false in the release record.

## Explicit limitations

The pilot is small, CPU-bound, and tied to declared fixtures. Its validation improvement is evidence that the implemented objective and optimization path can learn this bounded pilot; it is not evidence of general language ability or superiority over production Transformer systems. The control implementations are native reference controls for this gate, not optimized industrial frameworks. The checkpoint stores a deterministic seed/state label rather than a full external entropy-source replay because the pilot data order is deterministic. Scaling, distributed recovery, accelerator performance, large-corpus representativeness, privacy completeness, safety behavior, and human evaluation remain future gated work.
