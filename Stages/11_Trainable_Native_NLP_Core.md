# Stage 11 — Trainable Native NLP Core
## Real Next-Token Modeling, Checkpoints, and Matched Controls

**Predecessor:** Stage 10 — Tokenizer and Representation Engine  
**Successor:** Stage 12 — Scaling and Accelerator Systems  
**Status:** Specification; implementation not started  
**Implementation:** Native C++20 trainer, optimizer, checkpoints, tests, and gate

## Purpose

Stage 11 replaces the Stage 5 tiny surrogate with a real trainable next-token language objective around the CCT sequence core. It establishes whether CCT can optimize a language model on governed corpus shards while preserving the deterministic, checkpointable, and measurable behavior required for production engineering.

## Scope and non-goals

The stage includes token-batch loading, embeddings, output projection, causal next-token loss, masking, backpropagation through the CCT recurrence, optimizer state, gradient clipping, learning-rate schedules, validation, checkpointing, resume, anomaly detection, and matched dense Transformer/GRU/diagonal-SSM controls. It does not include large-scale distributed training, instruction tuning, preference alignment, retrieval, or production serving.

## Model contract

The base model must expose:

```text
forward(token_ids, masks, initial_state) -> logits, final_state, trace
loss(logits, targets, masks) -> scalar, token_count
backward(loss) -> gradients
optimizer_step(gradients) -> updated_parameters
checkpoint() -> weights, optimizer, scheduler, RNG, data_cursor, config, tokenizer_hash
resume(checkpoint) -> equivalent training state
```

The CCT candidate must use its selective recurrent state as the temporal backbone. It may include token embeddings, input/output projections, RMS normalization, complex state, causal-event features, or memory hooks only when the configuration records them. The matched baselines must use equal tokenizer, context length, training tokens, parameter-count band, optimizer budget, seeds, and hardware class.

## Required implementation

| Component | Implementation | Contract |
|---|---|---|
| Dataset reader | Sharded, resumable native reader | Cursor and shard identity are checkpointed |
| Batch builder | Packed/padded causal batches | Loss masks are exact |
| Embedding/output | Trainable input embedding and tied/untied head | Dimensions and parameter count are recorded |
| CCT forward | Streaming recurrent path and optional scan path | Chunked equivalence is tested |
| Loss | Stable cross-entropy with ignore masks | Non-finite values fail the run |
| Backpropagation | Analytic gradients through model and recurrence | Finite-difference spot checks pass |
| Optimizer | AdamW-equivalent baseline plus configured alternatives | Hyperparameters are serialized |
| Scheduler | Warmup/decay and step counter | Resume preserves schedule exactly |
| Stability | Gradient clipping, loss scaling, anomaly detection | Divergence is logged and stops safely |
| Checkpoint | Weights, optimizer, RNG, cursor, config, tokenizer, metrics | Resume is deterministic/tolerance-bounded |
| Evaluation | Validation loss, perplexity, throughput, memory | Held-out data is isolated |
| Baselines | Transformer, GRU, diagonal/SSM controls | Matched budgets are enforced |

## Training protocol

Every experiment must declare model configuration, tokenizer hash, corpus release, train/validation split, random seed, optimizer, learning rate, schedule, batch size, context length, target token count, parameter count, hardware, precision, and stopping rule. The trainer must emit periodic checkpoints and a final immutable artifact. A failed run must retain its last valid checkpoint and failure report.

The first pilot should use multiple small models and at least three seeds. It must report loss against tokens and wall-clock compute rather than reporting only final loss. A model that improves training loss but diverges on validation or produces non-finite logits fails.

## Evaluation harness

The native harness must test:

1. token-loss and gradient correctness on a tiny hand-computable fixture;
2. causal masking and packed-boundary correctness;
3. streaming/reference and chunked equivalence;
4. finite-difference gradient agreement on selected parameters;
5. optimizer update direction and schedule serialization;
6. deterministic seed initialization;
7. checkpoint resume equivalence after interruption at several cursor positions;
8. non-finite loss/gradient detection;
9. overfit capability on a tiny repeated corpus;
10. validation-loss improvement on a held-out corpus;
11. matched baseline parameter and compute accounting;
12. throughput, peak memory, and state-memory measurement;
13. reproducibility across at least three seeds.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Objective | Cross-entropy and perplexity are finite and correctly masked |
| Optimization | Three-seed pilot reduces validation loss relative to initialization |
| Gradient | Analytic/finite-difference spot checks pass declared tolerance |
| Stability | No unexplained divergence, NaN, or Inf in accepted runs |
| Checkpoint | Interrupted/resumed and uninterrupted runs agree within tolerance |
| Data cursor | Resume does not duplicate or skip records beyond declared policy |
| Baselines | All matched controls complete under the same declared budget |
| Capability | CCT beats the no-training control on next-token validation |
| Regression | Prior Stages 0–10 gates remain green |
| Efficiency | Tokens/sec, memory, state size, and compute are recorded |
| Reproducibility | Seed/config/corpus reruns are within declared variance |
| Artifact integrity | Checkpoint, tokenizer, config, data, and metrics hashes are complete |

## Pass/fail transition

Stage 11 passes only when a real next-token CCT model trains stably, improves held-out loss, resumes correctly, and is compared fairly with the matched baselines. A `PASS` authorizes Stage 12 scaling and accelerator work. It does not authorize instruction tuning or production use.

A `FAIL` requires optimizer, architecture, data, or numerical remediation. A `BLOCKED` result is valid if the available hardware cannot support the declared pilot or if data rights prevent training.

## Deliverables

The stage must deliver the native trainer, optimizer and scheduler, checkpoint V2+ format, dataset reader, model configuration, matched baseline runners, training reports, seed comparison, gradient report, resume report, resource profile, regression suite, gate executable, and CI command.

## Explicit limitations

A small next-token pilot does not prove broad language competence or scale efficiency. Loss and perplexity do not prove factuality, safety, instruction following, grounding, or production usefulness. Those require later stages and independent evaluations.
