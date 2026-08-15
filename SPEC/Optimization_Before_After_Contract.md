# Optimization Before/After Audit Contract

## 1. Purpose

This contract freezes the pre-optimization CCT implementation and defines the exact conditions under which an optimized result may be compared with it. No architecture, dataset, tokenizer, loss definition, or benchmark field may change between the before and after runs except where the specific upgrade is the subject of the measurement.

## 2. Frozen baseline

- Repository: `nexuss0781/CCT`
- Baseline commit: `a652618f17cbda92e0b81bc36ba224cfba92c4`
- Language: native C++20
- Build: CMake Release, strict warnings enabled
- Host identity: x86-64 Intel Xeon virtual CPU, six online CPUs
- Tokenizer snapshot hash: `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a`
- Vocabulary mode: compact
- Active vocabulary size: 513 slots
- Seed: 1701
- Data: repository’s governed real text artifacts with fixed train, validation, and test identities

## 3. Primary comparison contract

| Field | Frozen value |
|---|---:|
| Context length | 128 |
| Embedding dimension | 16 |
| Hidden dimension | 16 |
| Optimizer steps | 50 |
| Batch size | 4 |
| Training sequences | 64 |
| Evaluation sequences | 32 |
| Decode length | 256 tokens |
| Timing repeats | 20 |
| Optimizer | Adam-style contract already in `NlpOptimizerConfig` |
| Precision | Baseline `double`; after-run precision must be declared explicitly |

The four model kinds are evaluated independently under the same contract:

- Track 1 CCT recurrence.
- GRU.
- Diagonal SSM.
- Dense causal attention.

## 4. Required before/after measurements

Each model must report:

- Parameter count and parameter bytes.
- Optimizer-state bytes for first and second moments.
- Activation and scratch-buffer peak bytes.
- Peak resident process bytes.
- Checkpoint payload bytes and write/read elapsed time.
- Training target tokens, wall seconds, CPU seconds, and tokens/sec.
- Forward-only and backward-only wall time where separable.
- Evaluation tokens/sec and validation/test cross-entropy, perplexity, and token accuracy.
- Prefill milliseconds, first-token milliseconds, p50/p95/p99 inter-token latency, and generated tokens/sec.
- Batch-size and worker-count scaling.
- Deterministic output digest for a fixed prompt packet.
- Gradient equivalence and incremental-state equivalence results.

The benchmark must distinguish **model tokens**, **decoded text tokens**, **generated model tokens**, and **requests**. It must not combine these units without an explicit field name.

## 5. Correctness gates

The optimized path cannot be accepted if any of the following fails:

1. Reference and optimized forward logits agree within the declared numerical tolerance on all four architectures.
2. Reference and optimized gradients agree within the declared tolerance on finite-difference and analytic checks.
3. Incremental recurrent decoding agrees with full-context decoding for the same model state and token sequence.
4. Dense KV-cached decoding agrees with uncached dense decoding.
5. Fixed-seed training is deterministic within the declared tolerance.
6. Checkpoint save/load preserves parameters, optimizer state, tokenizer identity, dataset identity, lineage, and output digest.
7. Non-finite values, invalid dimensions, missing state, and corrupted checkpoints remain fail-closed.

## 6. Performance gates

Performance gains must be reported per architecture and per workload. A global pass cannot hide a regression in one model kind.

- Allocation-free or allocation-bounded hot loops must be demonstrated by instrumentation.
- Training and evaluation throughput must not regress under the primary contract unless the report identifies a deliberate quality-preserving trade-off.
- Recurrent decode must no longer recompute the entire prefix for every token.
- Attention decode must use a declared KV-cache policy.
- Worker-count changes must either affect execution and be measured or the worker setting must be removed from the public contract.
- Validation overhead must be measured separately from optimization throughput.
- Any lower-precision path must include loss, gradient, and output-quality comparisons against the double reference.

## 7. Training-eligibility gate

CCT training remains paused until all correctness gates pass, all four architecture paths have before/after reports, the resource accounting is complete, the incremental inference path is validated, and no P0 issue from the optimization audit remains open.
