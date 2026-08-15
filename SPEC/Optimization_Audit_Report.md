# CCT Optimization, Throughput, and Training-Eligibility Audit

**Author:** Manus AI
**Repository:** `nexuss0781/CCT`
**Audit checkout:** commit `6c2a4620b3a63e6fc4f03419273aa1dad2db2414`
**Date:** 2026-08-15
**Status:** **Training not eligible pending optimization and measurement upgrades**

## 1. Executive conclusion

The decision to terminate the continual-learning run was correct. The present CCT implementation is a valid native C++ research prototype, but it is **not yet eligible for further language training as an efficient, scalable, or production-oriented learning engine**.

The central problem is not one isolated bug. It is a stack of interacting limitations:

- The live trainer performs scalar, allocation-heavy forward and backward computation.
- The recurrent inference path recomputes the complete context for every generated token instead of advancing a persistent recurrent state.
- The output projection and softmax are evaluated over the full vocabulary at every supervised position, with untied input and output weights.
- The optimizer copies full parameter and moment vectors and recomputes Adam bias corrections inside the parameter loop.
- The curriculum trainer validates the entire validation set after every training step.
- The trainer is single-threaded even though the scaling configuration exposes a `worker_count` field.
- The current SFT preparation flattens OpenAssistant records to `role: text` lines; it does not preserve a structured prompt/response conversation with a response-only loss mask.
- The checkpoint-backed production inference service supports only the CCT model kind and has no incremental recurrent-state API or attention KV cache.
- The current metrics do not cover p50/p95 decode latency, allocation counts, activation memory, thread scaling, GPU execution, or model-token versus text-token unit consistency.

The small benchmark does show a useful architectural signal: **the CCT recurrence has lower state memory than dense attention and is faster than the current scalar dense-attention reference at the tested pilot dimensions**. That is not enough to establish superiority over optimized Transformer or state-space implementations. In the stable repeated benchmark, the diagonal SSM trains 19.08% faster than CCT, while CCT decodes 2.87% faster than the diagonal SSM and 7.52% faster than the GRU. The existing diagonal SSM is itself only a simple baseline, not an S4- or Mamba-equivalent implementation.

No additional Colab language training should begin until the mandatory optimization gates in Section 11 are green.

## 2. Audit scope and evidence contract

The audit examined the native C++20 build graph, the shared NLP trainer, all four live model kinds, tokenizer and batching code, curriculum preparation and session training, checkpoint serialization, the inference service, the scaling backend, and the architecture qualification harness. A new native executable, `cct_optimization_benchmark`, was added to the local research branch to separate training, evaluation, prefill, decode, and end-to-end generation measurements. It was compiled with the repository’s strict warning policy.

All comparative measurements used the repository’s real text artifacts, the pinned tokenizer snapshot, compact vocabulary mode, one CPU process, and the same model dimensions and optimizer contract across the four model kinds. The primary stable contract was:

| Measurement field | Value |
|---|---:|
| Active vocabulary slots | 513 |
| Context length | 128 |
| Embedding dimension | 16 |
| Hidden dimension | 16 |
| Training steps | 50 |
| Batch size | 4 |
| Training target tokens processed | 25,400 per model |
| Evaluation target tokens | 4,064 |
| Decode timing repeats | 20 |
| Generated tokens per decode | 256 |
| Tokenizer snapshot hash | `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a` |

The supporting JSON reports and the gprof profile are under `artifacts/optimization/`. The benchmark source is `cpp/tools/optimization_benchmark.cpp`.

> These measurements describe this implementation and this host. They are not claims about the asymptotic or production performance of CCT, GRU, SSM, or Transformer families in general.

## 3. Direct CCT comparison with the implemented baselines

### 3.1 Stable repeated benchmark

| Model | Parameters | Parameter bytes | State bytes reported by model | Serialized model bytes | Train tok/s | Eval tok/s | End-to-end decode tok/s | Validation loss before → after | Test loss before → after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CCT recurrence | 18,001 | 144,008 | 256 | 375,607 | 39,361.90 | 85,286.88 | 1,765.37 | 6.3457 → 5.9895 | 6.3804 → 6.0287 |
| GRU | 18,513 | 148,104 | 128 | 386,289 | 42,670.03 | 82,710.04 | 1,641.92 | 6.2829 → 6.1880 | 6.2815 → 6.1884 |
| Diagonal SSM | 17,201 | 137,608 | 128 | 359,211 | 46,871.74 | 84,345.59 | 1,715.97 | 6.2586 → 6.2074 | 6.2448 → 6.1934 |
| Dense causal attention | 17,697 | 141,576 | 32,896 | 369,323 | 35,103.00 | 26,218.28 | 331.20 | 6.2416 → 6.1993 | 6.2433 → 6.2029 |

Relative to CCT in this contract, the diagonal SSM trains **19.08% faster**, the GRU trains **8.40% faster**, and dense attention trains **10.82% slower**. CCT’s end-to-end decode rate is **7.52% higher than GRU**, **2.87% higher than the diagonal SSM**, and **433.03% higher than the current dense-attention reference**.

The last comparison must be interpreted carefully. The dense implementation is a scalar reference with repeated allocations and no KV cache or IO-aware tiling. It is not a fair comparison against an optimized Transformer runtime. The result identifies a large optimization opportunity in the dense baseline, not a proof that CCT beats Transformers.

### 3.2 Context-length sweep

The context sweep used 10 steps, batch size 4, width 16, 128 generated tokens, and five timing repeats.

| Context | CCT end-to-end tok/s | GRU end-to-end tok/s | Diagonal SSM end-to-end tok/s | Dense attention end-to-end tok/s |
|---:|---:|---:|---:|---:|
| 32 | 5,828.09 | 5,492.94 | 5,545.80 | 2,641.04 |
| 64 | 3,295.03 | 3,046.43 | 3,210.56 | 981.42 |
| 128 | 2,308.72 | 2,151.48 | 2,289.26 | 513.86 |
| 256 | 2,303.05 | 2,143.08 | 2,274.89 | 506.52 |

The recurrent models do not maintain constant decode throughput as context grows. At context 32, CCT reaches 5,828 tok/s; at context 128, it reaches 2,309 tok/s. The reason is visible in the code: `CheckpointInferenceBackend` calls `NextTokenModel::next_logits(context)` for each token, and `next_logits` runs a complete forward pass over the supplied context. A persistent CCT state would remove this repeated prefix computation.

Dense attention loses more sharply and its reported model state grows from 8,320 bytes at context 32 to 65,664 bytes at context 256. These values are model-state fields, not total process memory.

### 3.3 Width sweep

At context 128, increasing width reduces throughput rapidly because the implementation is dominated by scalar matrix-vector and vocabulary-head loops.

| Width | CCT parameters | CCT train tok/s | CCT eval tok/s | GRU train tok/s | SSM train tok/s | Dense train tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 18,001 | 29,232.72 | 84,107.71 | 30,148.70 | 31,915.78 | 17,909.66 |
| 32 | 37,537 | 17,693.99 | 55,486.63 | 11,699.76 | 14,320.13 | 7,356.32 |
| 64 | 82,753 | 8,453.17 | 29,671.38 | 8,307.86 | 11,123.80 | 3,357.23 |

The width results are short-run engineering measurements, so decode rates are more timer-sensitive than training rates. The robust structural result is that the current kernels do not scale gracefully with width, and the dense attention reference degrades fastest.

## 4. Root-cause code audit

### 4.1 P0: training-eligibility blockers

#### P0.1 The current language objective is not yet a conversation-learning objective

`curriculum_prepare.cpp` writes OpenAssistant examples as `role + ": " + text`, and `curriculum_session.cpp` reads each non-empty line as an independent document. The SFT phase therefore applies ordinary next-token loss to a flattened role-prefixed line. It does not preserve a multi-turn conversation tree, distinguish prompt tokens from response tokens, mask loss to assistant responses, or test whether a response follows an instruction. This is insufficient for the stated final objective of native-English conversational understanding and ambiguity handling.

The first repair must define a structured conversation format, deterministic role and turn separators, prompt/response loss masks, held-out conversation identities, and competency-specific tests. SFT should not be treated as complete merely because its cross-entropy decreases.

#### P0.2 The curriculum pays validation cost after every optimizer step

`curriculum_session.cpp:160-172` sets `validation_interval_steps = 1`. `NlpTrainer::train_step` then evaluates the entire validation split after every step. This makes the real curriculum throughput dependent on repeated full validation and prevents a clean separation between optimization speed and measurement speed. Validation should be performed at a declared interval, at epoch boundaries, and at the final checkpoint, with its cost reported separately.

#### P0.3 The live inference service is not architecture-neutral

`CheckpointInferenceBackend` in `cpp/src/inference.cpp:90-105` rejects every checkpoint whose kind is not `Track1CctRecurrence`. GRU, diagonal SSM, and dense attention therefore cannot be compared through the same production service. A benchmark that compares four trainer model kinds but deploys only CCT does not yet establish production parity.

#### P0.4 The current output is not language-competent evidence

The measured tiny runs reduce held-out loss, but the architecture qualification generations are repetitive and contain invalid-looking byte output for all four models. This is expected at width 16 and a short training horizon, but it means the system is not language-training eligible under the project’s standard. The benchmark must separate finite optimization from usable English generation and must not advance the curriculum from loss reduction alone.

#### P0.5 Current memory reporting is incomplete

`NextTokenModel::state_memory_bytes()` reports only a model-specific recurrent or attention state estimate. It omits parameter storage, optimizer moments, activations, gradient buffers, temporary vectors, allocator overhead, tokenizer/data memory, checkpoint buffers, and service queues. `ScalingRunner` reports peak resident memory, but the core architecture qualification report does not. Any resource claim must report these categories separately.

### 4.2 P1: dominant performance blockers

#### P1.1 Allocating matrix-vector kernels dominate the CPU profile

The gprof profile recorded `matvec` at **19.77% self time across 5,340,224 calls**. The helper allocates a new `std::vector<double>` for every matrix-vector result and performs scalar nested loops. The forward paths call it repeatedly inside token and gate loops. This creates allocator traffic, poor cache reuse, and no explicit vectorization.

Required repair: introduce contiguous tensor views and `matvec_into`/fused kernels that write into caller-owned scratch buffers. Reuse per-sequence and per-batch storage. Add correctness tests comparing fused and reference kernels.

#### P1.2 Softmax and full vocabulary projection are repeated at every position

`softmax` consumed **13.75% self time across 170,688 calls** in the profile. Every time step constructs a full vocabulary logits vector, then constructs a full probability vector during loss computation. The output head loops over every vocabulary slot and hidden coordinate for every position. This is appropriate for a reference implementation but not for efficient scale.

Required repairs include contiguous logits buffers, fused cross-entropy with log-sum-exp, reuse of probability scratch when probabilities are actually needed, optional sampled/adaptive vocabulary objectives for large vocabularies, and tied input/output embeddings when dimensions permit.

#### P1.3 Forward functions materialize all time-step logits for one-token inference

`NextTokenModel::next_logits` creates an `NlpSequence`, calls `model_forward`, and retains a complete `vector<vector<double>>` for the whole context even though only the last logits are used. This is a direct inference waste. A `forward_last_logits` or incremental `step` API should return only the final projection and preserve the recurrent/attention state.

#### P1.4 Recurrent inference recomputes the context from zero

CCT, GRU, and diagonal SSM all start from a zero hidden state inside their forward functions. The production path therefore recomputes all prior tokens for every generated token. This explains the context sweep degradation and defeats the principal online-state advantage of recurrence.

Required repair: expose a serializable `RecurrentState`, a `reset_state`, a `consume_token`, and a `next_logits_from_state` API. The service must maintain the state per conversation and enforce state-version, tokenizer, model, and context identities.

#### P1.5 Dense attention recomputes keys, values, and attention history

`forward_dense` recomputes key and value projections for every prior position at every time step. `dense_gradients` contains quadratic position loops and materializes per-position vectors. A fair optimized baseline requires a KV cache for decoding and a tiled/fused attention kernel for training. The current dense result is useful as a scalar reference only.

#### P1.6 Optimizer updates copy full model state

`NlpTrainer::train_step` creates `parameters`, `next_first_moment`, and `next_second_moment`, then creates `candidate_model = model_`, and calls `set_parameter_vector`. This copies full parameter and moment arrays on every step. The transactional safety goal is legitimate, but the implementation should stage updates in reusable buffers or commit in place after finite checks.

#### P1.7 Adam bias correction is inside the parameter loop

`first_correction` and `second_correction` are independent of the parameter index but are computed with `std::pow` inside the loop over every parameter. Compute them once per optimizer step. A fused AdamW kernel should update parameters and moments in one contiguous pass, with vectorized arithmetic and no temporary model copy.

#### P1.8 The live trainer is not batched at the tensor-kernel level

`train_step` loops over batch elements and calls the single-sequence `loss_and_gradients` path separately. It aggregates gradients only after each independent computation. There is no batch-major input tensor, no shared recurrent kernel over batch lanes, and no parallel worker pool. The `worker_count` field in `ScalingPointConfig` is validated but not used by `ScalingRunner::run`.

#### P1.9 The implementation does not use available CPU parallelism or vector units

The host exposes six CPUs and AVX2/AVX-512 capability, but the live trainer is scalar and single-threaded. The CMake file enables strict warnings but does not define an explicit architecture-tuned kernel layer, OpenMP/TBB execution policy, BLAS/oneDNN backend, or portable SIMD abstraction. This is not a reason to add unsafe `-ffast-math`; numerical semantics must be tested first.

#### P1.10 Tokenization uses a linear candidate scan

`Tokenizer::encode_content` scans `piece_order_` from the beginning at every byte position and compares candidate strings. The gprof profile recorded `Tokenizer::encode_content` at **12.89% self time** in the benchmark process. Data preparation is not the primary training hot loop, but this design becomes expensive when real corpus volume increases.

Required repair: construct a trie or compact prefix index for greedy longest-match tokenization, preserve the snapshot hash contract, and benchmark byte throughput and allocation rate.

#### P1.11 Dataset records and sequences are over-materialized

`curriculum_session.cpp` reads every non-empty line into `std::vector<std::string>`, tokenizes every record into `EncodedDocument`, then `NlpDataset::build` copies token IDs into per-sequence vectors. The existing `CausalBatchPacker` provides contiguous packed and padded representations, but the live NLP trainer does not use them. Large-scale training should use packed immutable token arrays, sequence offsets, and reusable batch views rather than a graph of small vectors and strings.

#### P1.12 Chunking discards cross-chunk next-token transitions

`NlpDataset::build` slices each document into non-overlapping context chunks and masks the last position of every chunk. The next token across a chunk boundary is therefore not trained. This is a data-efficiency defect, not only a performance defect. A packed stream with controlled document boundary markers or a stride/overlap policy should preserve valid causal transitions without allowing train/validation leakage.

#### P1.13 Checkpoints are text decimal streams

`save_checkpoint` writes model parameters and optimizer moments as text with decimal doubles. This inflates checkpoint size, increases serialization CPU time, and makes large-scale persistence expensive. The atomic publication behavior should be retained, but payloads should use a versioned binary format with explicit endianness, dtype, counts, checksums, and optional compression.

### 4.3 P2: architectural and measurement limitations

#### P2.1 CCT is currently a small gated recurrence, not yet a demonstrated new scaling regime

The CCT update uses retain, write, candidate, and previous-input effects. At the tested configuration it has 18,001 parameters, only slightly more compact than the SSM and GRU baselines. It does not yet contain a hardware-aware scan, content-selective state parameterization, multi-layer residual path, normalization strategy, or explicit mechanism for preserving multiple timescales. The architecture may become valuable, but the current code does not establish a new class of capability.

#### P2.2 The diagonal SSM baseline is not S4 or Mamba

The live diagonal SSM keeps one learned retain scalar per hidden coordinate and applies a fixed `0.999` factor. S4 uses structured state-space parameterization and efficient kernels, while Mamba adds input-dependent selectivity and hardware-aware recurrent-mode computation [2] [3]. Therefore, CCT’s comparison is currently against a deliberately simple diagonal recurrence, not against the strongest state-space engineering baseline.

#### P2.3 The Transformer baseline is not optimized

The dense attention implementation is a useful correctness control but has no KV cache, no tiling, no fused softmax, and no IO-aware memory strategy. FlashAttention demonstrates that practical attention speed depends on memory traffic and tiling rather than only the quadratic formula [4]. The current CCT-versus-attention gap must not be advertised as a production Transformer win.

#### P2.4 Benchmark counters are not fully comparable

The architecture qualification counter `target_tokens_per_second` includes training-step wall time and any validation executed inside `train_steps`, while the evaluation counter measures only held-out evaluation. The inference service reports input/output counts using a whitespace-based `token_count` for fixture paths, while the checkpoint backend counts tokenizer tokens. The report must label model-token, text-token, and generated-token units separately.

#### P2.5 Training and inference lack p50/p95 stability data

The current benchmarks report aggregate seconds and averages. They do not report warmup runs, CPU affinity, repeated-run median, p50/p95/p99 per-token latency, allocator counts, cache effects, or concurrent request throughput. One short timer measurement is not enough for a production resource claim.

## 5. Architecture-family comparison

The external literature provides useful design principles but does not validate CCT. The original Transformer paper emphasizes parallelizable sequence computation and reports a departure from recurrence [1]. GRU-style recurrence offers a compact online state but retains sequential dependence and a fixed state bottleneck [5]. S4 demonstrates that structured state-space parameterization can make long-sequence computation practical [3]. Mamba adds input-dependent selection and hardware-aware recurrent-mode execution, addressing content selectivity and systems efficiency [2]. FlashAttention shows that IO-aware tiling can materially change the wall-clock behavior of attention [4].

| Family | Training parallelism | Decode state | Long-context scaling | Current CCT-code comparison |
|---|---|---|---|---|
| Dense attention | High over positions, but quadratic attention work | KV cache required for efficient decode | Quadratic reference path; optimized kernels alter practical cost | Scalar, no KV cache, no tiling; control baseline only |
| GRU | Sequential recurrence | Compact hidden state | Linear recurrence but fixed state capacity | Implemented and slightly faster to train in stable pilot |
| Diagonal SSM | Sequential or scan-friendly | Compact hidden state | Linear in simple recurrence | Implemented, trains 19.08% faster than CCT in pilot |
| S4-style SSM | Structured scan/convolution | Structured state | Designed for long sequences | Not implemented |
| Mamba-style selective SSM | Hardware-aware selective recurrence | Compact recurrent state | Linear-time design with content selectivity | Not implemented |
| Current CCT | Sequential gated recurrence | Compact conceptual state | Code path recomputes context during decode | Not yet a fused or incremental implementation |

## 6. Major upgrade program

### Upgrade 1 — Establish a trustworthy measurement layer

Implement a native benchmark contract with warmup runs, repeated medians, p50/p95/p99 latency, model tokens/sec, generated tokens/sec, input tokens/sec, CPU seconds, resident memory, parameter memory, optimizer memory, activation memory, allocation counts, and thread count. Pin or record CPU affinity, compiler, build type, CPU model, SIMD flags, and dataset/tokenizer hashes. The benchmark must separately measure prefill, incremental decode, training forward/backward, optimizer update, validation, checkpoint write, and checkpoint reload.

### Upgrade 2 — Replace allocating scalar kernels

Introduce a small native tensor-kernel layer using contiguous storage and non-owning views. Add `matvec_into`, batched matrix-vector products, fused gate kernels, reusable scratch arenas, and explicit alignment. Implement reference and optimized paths and compare their outputs within a declared numerical tolerance. Use portable SIMD or an optional CPU backend; do not silently change floating-point semantics.

### Upgrade 3 — Implement real batch training

Represent batches as contiguous `[batch, time, width]` or equivalent layouts with sequence offsets and masks. Parallelize independent batch lanes and fuse shared projection work. Make `worker_count` functional or remove it from the contract. Measure strong scaling from one to the available CPU workers and report reproducibility across worker counts.

### Upgrade 4 — Make optimization memory-safe and efficient

Move to in-place or reusable-buffer AdamW. Precompute bias-correction scalars once per step. Fuse moment updates and parameter updates. Add optional FP32 parameters/moments with FP64 diagnostic mode, and test loss/gradient parity. Tie input embeddings to the output projection when dimensions permit. For larger vocabularies, evaluate adaptive or sampled vocabulary objectives with a held-out full-softmax evaluation path.

At the current 513-slot, width-16 pilot, tying the embedding and output weights would remove 8,208 parameters from the 18,001-parameter CCT model, approximately a 45.6% reduction, subject to a correct tied-head design. At realistic vocabularies, the larger benefit is memory: parameters and two Adam moment arrays currently consume approximately 24 bytes per parameter in double precision before activations and temporary buffers.

### Upgrade 5 — Implement incremental inference

Add a model-neutral interface with `reset_state`, `prefill`, `step`, `state_bytes`, and `serialize_state`. For CCT, GRU, and SSM, each token should advance the state once and project only the next logits. For attention, add a KV cache and a tiled attention path. Replace front-erasing vectors with ring buffers. Maintain exact model/tokenizer/state identities in service snapshots.

### Upgrade 6 — Correct the language-learning objective

Keep pretraining as a pure next-token objective. Redesign SFT around structured conversation records with deterministic role markers, turn boundaries, prompt/response masks, response-only loss, and held-out conversation identities. Add separate evaluation suites for sentence completion, local grammar, instruction following, ambiguity recognition, clarification quality, continuity, and repair. The curriculum must measure learning behavior, not only cross-entropy.

### Upgrade 7 — Upgrade the architecture only after the kernel baseline is fair

After the reference kernels and incremental state path are optimized, compare three controlled CCT variants:

1. **CCT-Select:** input-dependent retain/write parameters for content-selective memory.
2. **CCT-Scan:** fused parallel scan or chunkwise associative recurrence for training throughput.
3. **CCT-Hybrid:** a compact recurrent state plus bounded local content-mixing path to recover some content-addressing ability without full quadratic attention.

Each variant must be compared with a real GRU, a stable structured SSM baseline, and an optimized attention control under identical data, width, token budget, precision, and hardware. The result must include both quality and resource Pareto fronts.

## 7. Required qualification gates before training resumes

| Gate | Required evidence | Pass condition |
|---|---|---|
| Measurement integrity | Repeated native benchmark with host/compiler/data identity | Metrics are unit-labeled, repeatable, and include p50/p95/p99 where applicable |
| Kernel correctness | Reference-versus-optimized forward and gradient tests | Declared numerical tolerance passes for every architecture |
| Training efficiency | Batch and worker scaling report | Throughput improves or remains stable as workers and batch size increase; no hidden serial bottleneck |
| Memory accounting | Parameter, optimizer, activation, temporary, resident, and checkpoint bytes | All categories are reported and bounded against an explicit budget |
| Incremental inference | Prefill/decode benchmark with persistent state | Recurrent models do not recompute the complete prefix per output token |
| Attention fairness | KV-cache and fused/tiled control | CCT is not compared against an intentionally unoptimized attention straw man |
| Objective validity | Structured SFT and masked response loss audit | Prompt, response, roles, and held-out identity are preserved |
| Language sanity | Unseen generation and competency suites | No invalid-byte collapse, EOS collapse, or repetitive degeneration at the declared level |
| Continual-learning integrity | Parent lineage, disjoint data, retention and forgetting report | Every accepted session improves its target without unacceptable regression on prior competencies |
| Production eligibility | Full inference-service model support and SLO report | Runtime route, token units, latency, failure behavior, and state quotas are all tested |

Until these gates pass, the status remains **optimization required; training paused**.

## 8. Recommended implementation order

1. Freeze the current benchmark artifacts and preserve the current checkpoint lineage as a baseline.
2. Finish the native measurement layer and correct token-unit accounting.
3. Replace allocating kernels and add reference-equivalence tests.
4. Implement real batch/worker execution and measure scaling.
5. Add in-place fused optimizer updates and binary checkpoints.
6. Add incremental CCT/GRU/SSM inference and a fair attention KV-cache control.
7. Redesign structured SFT and competency evaluation.
8. Re-run the four-model Pareto benchmark.
9. Only then implement CCT-Select, CCT-Scan, or CCT-Hybrid architectural upgrades.
10. Resume the continual-learning curriculum only if the eligibility gates are green.

## 9. Current verdict

**CCT is not currently eligible for further Colab training.** The immediate objective is not another dataset chunk or another Module 1 session. The immediate objective is to make the engine measurable, memory-conscious, batch-capable, incrementally decodable, and aligned with the intended conversational learning objective.

The current evidence supports one limited positive statement:

> The CCT recurrence has a compact state representation and, in this scalar pilot, outperforms the current dense-attention reference in measured decode throughput.

The evidence does **not** support claims that CCT is more efficient than optimized Transformers, S4, Mamba, or production GRUs; that it is ready for broad language training; or that it has acquired English or conversational ambiguity competence.

## References

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need — Vaswani et al."

[2]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces — Gu and Dao"

[3]: https://arxiv.org/abs/2111.00396 "Efficiently Modeling Long Sequences with Structured State Spaces — Gu, Goel, and Ré"

[4]: https://arxiv.org/abs/2205.14135 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al."

[5]: https://arxiv.org/abs/1406.1078 "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation — Cho et al."
