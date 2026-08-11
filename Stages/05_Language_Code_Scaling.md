# Stage 5 — Language and Code Scaling

**Project:** CCT-ASE  
**Stage ID:** 5  
**Predecessor:** Stage 4 — Persistent Verifiable Memory  
**Successor:** Stage 6 — Deliberation and Verification  
**Status:** Specification; implementation not started

## Purpose

Stage 5 scales the validated CCT-ASE core to language and code while preserving the recurrence, causal-event, and memory contracts. The central question is whether the architecture provides a favorable quality–compute–memory trade-off on realistic data, not whether it can produce fluent text.

The stage must compare CCT-ASE with matched Transformer, recurrent, and state-space baselines under controlled data, token, parameter, training-compute, and inference-latency budgets. It must also establish data provenance, contamination controls, checkpoint reproducibility, and code-execution evaluation.

## Scope and non-goals

The stage includes tokenizer or event vocabulary decisions, licensed and provenance-tracked corpora, deduplication, data mixture configuration, scalable training, checkpointing, language modeling, code modeling, long-context evaluation, memory-augmented training, and model documentation. It does not add autonomous tool use, unrestricted internet access, self-modification, or open-ended agent deployment.

The initial scale should be deliberately small enough to support repeated ablations. Large-scale training is allowed only after the small-scale scaling trend is reproducible and no mandatory Stage 4 memory or Stage 2 sequence gate has regressed.

## Data and governance contract

Every training and evaluation sample must carry dataset, source, license, split, hash, filtering, and contamination metadata. The data pipeline must support deterministic replay from a manifest.

| Data property | Required implementation | Failure condition |
|---|---|---|
| Provenance | Record source, license, collection date, and transformation history | Sample enters training without provenance |
| Deduplication | Apply exact and near-duplicate filtering with audited thresholds | Evaluation text is duplicated in training |
| Split integrity | Keep train, validation, and test manifests immutable after release | Split changes after results are observed |
| Sensitive content | Apply documented filtering and access controls | Sensitive data is used without review |
| Code licensing | Track repository/license metadata and exclude incompatible sources | Code corpus cannot be legally or technically audited |
| Contamination | Maintain held-out canaries and overlap checks | Benchmark leakage is unknown |
| Reproducibility | Persist manifest hash and sampling seed | Training data cannot be reconstructed |

The evaluation set must remain inaccessible to training jobs. Benchmark answers, hidden test files, and evaluator-only labels must be stored outside the training data path.

## Model and training contract

The model must support:

```python
loss, metrics = model.forward(batch, state=None, memory=None)
state = model.step(event_or_token, state, memory=None)
checkpoint = trainer.save(model, optimizer, scheduler, rng, data_cursor)
```

The training system must implement next-token or next-event prediction, masked or span objectives only when justified by the model design, mixed precision, gradient accumulation, learning-rate scheduling, checkpoint resume, activation rematerialization, distributed data loading, and metric logging.

Memory augmentation must be switchable. The base model, base model plus frozen memory, and jointly trained memory controller must be separate configurations. Any retrieved context must be logged with record IDs and must not be counted as model parametric knowledge.

A recommended total objective is:

```text
L = L_next + λ_mem L_retrieval + λ_state L_consistency
  + λ_evidence L_citation + λ_stab L_stability
```

All terms must be logged separately. The final training loss alone is not an acceptable success metric.

## Required implementation

| Component | Required implementation | Acceptance artifact |
|---|---|---|
| Vocabulary | Versioned tokenizer/event vocabulary with unknown and byte fallback behavior | Vocabulary hash and round-trip tests |
| Data loader | Sharded, deterministic, resumable loader with sequence packing | Cursor-resume equivalence report |
| Trainer | Single-device reference trainer before distributed optimization | Reproducible micro-run |
| Optimizer | Declared optimizer, schedule, clipping, weight decay, and precision policy | Config and optimizer-state report |
| Distributed path | Data parallelism first; sharding only after reference correctness | Single-versus-multi-device equivalence on a small run |
| Checkpoints | Store model, optimizer, scheduler, RNG, data cursor, schema, and git commit | Resume reproduces metrics and next batch |
| Evaluation server | Isolated evaluator with fixed prompts/data and JSON results | Immutable evaluation artifact |
| Profiling | Measure FLOPs estimate, tokens/sec, memory, communication, compile time, and latency | Hardware profile per run |
| Model card | Document data, architecture, limitations, benchmarks, and known failure modes | Reviewable model card |

## Matched baseline protocol

At minimum, compare:

- A dense causal Transformer.
- A recurrent or GRU baseline.
- A plain selective state-space baseline.
- CCT-ASE without memory.
- CCT-ASE with memory.

Comparisons must be made at matched parameter count bands, training-token budgets, optimizer budget, and evaluation tokenization where possible. Report both quality at equal training compute and quality at equal inference latency. If a model uses retrieval or external memory, report memory-build cost, retrieval latency, and storage footprint separately.

## Evaluation harness

### Language modeling

Measure validation perplexity or cross-entropy on held-out corpora, broken down by document length, domain, and sequence position. Evaluate next-token calibration, entropy, repetition, and degradation under long sequences. Report both teacher-forced and free-running generation diagnostics.

### Long-context behavior

Use retrieval, needle-in-context, multi-document synthesis, and temporal ordering tasks with known evidence locations. Vary context length beyond the training length and compare recurrent state, memory retrieval, and chunked processing. A model must not receive credit for reproducing a memorized benchmark string without locating the relevant evidence when evidence attribution is part of the task.

### Code modeling and execution

Evaluate code completion, repository-level retrieval, unit-test generation, bug fixing, and program synthesis on held-out projects. Execute generated code in a sandbox with resource limits. Score exact tests passed, compilation, security policy violations, and failure explanations. Never execute generated code on the host or with unrestricted network access.

### Reasoning and generalization

Use algorithmic tasks, compositional instruction following, symbolic transformations, and held-out combinations. Report exact match, execution success, calibration, and length extrapolation. Reasoning claims must be separated from memorization and benchmark contamination.

### Memory-augmented evaluation

Compare no-memory, frozen-memory, learned-memory, and oracle-retrieval variants. Measure retrieval quality, answer quality conditional on retrieval correctness, citation quality, and failure when memory is stale or contradictory.

### Efficiency

Measure training tokens per second, wall-clock time to target validation loss, peak memory, decode latency, throughput, storage, and energy if available. Plot quality against compute and latency. Asymptotic complexity must be supported by raw scaling data.

### Robustness and data controls

Evaluate prompt variations, tokenization changes, noisy or missing metadata, document order changes, duplicated passages, adversarial memory content, and held-out domains. Run contamination checks against all public evaluation material.

## Pass/fail criteria

| Criterion | Pass condition | Failure condition |
|---|---|---|
| Data audit | Every training/evaluation shard is manifest-addressed and provenance-reviewed | Unknown or unlicensed data enters a reported run |
| Replayability | A micro-run and checkpoint resume reproduce loss, metrics, and data cursor | Resume changes trajectory without explanation |
| Model correctness | Batched, chunked, and recurrent decode paths preserve the Stage 2 contract | Long-sequence path silently changes state semantics |
| Language quality | CCT-ASE meets the predefined quality floor against matched baselines or demonstrates a declared efficiency advantage at equivalent quality | It is evaluated only by cherry-picked generations |
| Code quality | Generated code is measured by sandboxed execution and security checks | Syntax-only or subjective grading is used as the main metric |
| Long context | Performance curves and evidence localization are reported beyond training length | A single context length is reported |
| Memory value | Memory augmentation improves a declared metric without increasing unsupported claims | Gains come from leaked answers or untracked retrieval |
| Efficiency | Quality–compute–latency frontier is measured and no hidden quadratic hot path appears | Complexity claim lacks profile or raw scaling data |
| Contamination | Held-out overlap and canary tests pass | Benchmark contamination is unresolved |
| Robustness | Model degrades gracefully under order, noise, and stale-memory perturbations | Minor perturbations cause silent policy or state failure |
| Documentation | Model card records data, training, limitations, evaluations, and known failures | Results cannot be independently interpreted |

Stage 5 does not require CCT-ASE to beat every Transformer. It requires an honest, reproducible comparison and at least one material advantage—quality at lower compute, lower memory at equal quality, better long-context retention, or better evidence integrity—without a mandatory safety or correctness regression.

## Transition to Stage 6

Stage 6 may begin when a named CCT-ASE checkpoint has passed data, reproducibility, language, code, long-context, memory-integrity, and efficiency gates. The checkpoint must be frozen for the first deliberation experiments so that improvements from the new workspace can be measured.

The transition package must include manifests, hashes, training logs, checkpoint metadata, baseline configurations, contamination report, code-sandbox report, quality–compute curves, long-context curves, model card, and an explicit list of capabilities not demonstrated.

If the stage fails, the team must not immediately increase model size. First determine whether the cause is data quality, optimization, tokenizer, recurrence, memory retrieval, or evaluation leakage. Any revised run must receive a new manifest and checkpoint identity.

## Exit report

The report must distinguish parametric knowledge, retrieved knowledge, generated reasoning, and executed code. It must include confidence intervals or seed variance for key results and must identify tasks where CCT-ASE remains inferior to baselines.

**Transition decision:** `PASS` authorizes Stage 6. `FAIL` requires remediation. `BLOCKED` is allowed only for optional large-scale distributed training; the small-scale reproducible language/code suite must pass.

## References

[1]: ../CCT_EVOLUTION_PROPOSAL.md "CCT-ASE evolution proposal"

[2]: ../Stages/02_Sequence_Core.md "CCT Stage 2 sequence core specification"

[3]: ../Stages/04_Persistent_Verifiable_Memory.md "CCT Stage 4 persistent memory specification"
