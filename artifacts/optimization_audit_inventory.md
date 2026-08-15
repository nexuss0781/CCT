# CCT Optimization Audit — Initial Inventory and Benchmark Evidence

## Audit identity

- Repository: `nexuss0781/CCT`
- Checkout: local commit `6c2a4620b3a63e6fc4f03419273aa1dad2db2414`
- Language: native C++20
- Build: CMake, strict `-Wall -Wextra -Wpedantic -Werror`
- Host measurement environment: x86-64 Intel Xeon virtual CPU, 6 online CPUs, AVX2/AVX-512 flags visible, GCC C++20, FFTW3 available.
- Tokenizer snapshot hash: `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a`.

## Native execution map

The `cct_native` static library compiles corpus and tokenizer code, the shared NLP trainer, the four model kinds, inference, scaling, release, and the staged gates. The live model comparison executable is `cct_architecture_qualification`; the resource backend is `ScalingRunner`; the checkpoint-backed runtime is `CheckpointInferenceBackend`; the continual-learning tools are dataset preparation, session training, and checkpoint inspection.

The shared NLP trainer currently stores all parameters and Adam moments in `std::vector<double>`. Every architecture uses a vocabulary projection and bias. The compact vocabulary benchmark uses 513 active output slots. The model forward and gradient implementations are scalar nested loops with per-token temporary `std::vector<double>` allocations and repeated `matvec` calls.

## Matched architecture benchmark

Command contract:

```text
cct_architecture_qualification
  --context 128 --embedding 16 --hidden 16 --steps 20 --batch 4
  --train-sequences 64 --eval-sequences 32 --vocab-mode compact --seed 1701
```

Data: the governed Track 1 real-release FineWeb/WikiText-style text artifacts already in the repository. Training used 8,128 target tokens after sequence capping; validation and test each used 4,064 target tokens. Active vocabulary was 513 slots. The benchmark ran on one native process under the sandbox CPU.

| Model | Parameters | State bytes | Serialized model bytes | Train seconds | Target tokens/sec | Validation loss before -> after | Test loss before -> after |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCT recurrence | 18,001 | 256 | 375,629 | 0.273043861 | 37,210.13892 | 6.345714643 -> 6.191973627 | 6.380355094 -> 6.227254392 |
| GRU | 18,513 | 128 | 386,319 | 0.287895890 | 35,290.53506 | 6.282914294 -> 6.247344646 | 6.281545141 -> 6.246619178 |
| Diagonal SSM | 17,201 | 128 | 359,161 | 0.235749863 | 43,096.52557 | 6.258635719 -> 6.238950149 | 6.244817901 -> 6.225041106 |
| Dense causal attention | 17,697 | 32,896 | 369,290 | 0.403500021 | 25,179.67651 | 6.241580826 -> 6.224745756 | 6.243309277 -> 6.227167983 |

Relative to the measured CCT recurrence, the diagonal SSM is 15.8193% faster, the GRU is 5.1589% slower, and dense attention is 32.3312% slower. CCT is 47.7784% faster than dense attention in this benchmark. Dense attention’s reported state bytes are 12,750% larger than CCT’s recurrence state bytes at context 128.

## Initial interpretation boundaries

These are controlled native measurements, not production-scale claims. The harness reports training target tokens/sec only; it does not report hardware FLOPs, thread utilization, memory bandwidth, generation tokens/sec, p50/p95 latency, allocation counts, or CPU/GPU scaling. Generation diagnostics are qualitative and show repetitive/invalid-looking outputs for all four tiny models after the bounded run. The benchmark therefore establishes an optimization signal and a measurement gap, not language competence.

## Immediate code-level observations

1. `NlpTrainer::train_step` performs one `loss_and_gradients` call per sample in a batch, accumulates a full parameter-sized gradient, and then performs a second full parameter-sized Adam update.
2. Every model’s forward and backward path repeatedly allocates small vectors inside time loops and calls a generic allocating `matvec` helper.
3. The output head is evaluated by looping over all 513 output slots at every time step; the embedding and output head are untied, so vocabulary scaling costs both a large embedding matrix and a large head matrix.
4. The dense attention forward path recomputes key/value projections for all previous positions at every time step, and its backward path uses quadratic position loops.
5. Adam bias-correction powers are recomputed inside the parameter loop for every parameter on every optimizer step.
6. The trainer copies the entire parameter vector and both moment vectors on every update, and then copies the candidate model parameter vector again through `set_parameter_vector`.
7. Validation can run on every step, and the current comparison harness sets validation to the final step only; there is no explicit benchmark switch for validation overhead.
8. `NextTokenModel::evaluate` computes a full logits matrix for each sequence and materializes a probability vector for each supervised position; it does not batch sequences.
9. Checkpoint serialization is text-based with decimal doubles, which makes checkpoints large and slower to write/read than a binary representation.
10. The checkpoint-backed runtime currently hard-requires the Track 1 CCT recurrence model kind and performs a full `next_logits` call for each generated token after rebuilding/windowing a `std::vector` context.
11. The current runtime inference path has no KV cache for attention, no recurrent-state incremental API for CCT/GRU/SSM, and no batched inference path.
12. The benchmark’s `target_tokens_per_second` excludes initial/final evaluation but includes all training-step work and any validation performed during `train_steps`; it is not directly comparable to the scaling backend’s wall-clock metric unless the same contract is used.

## Next audit work

- Inspect all hot loops and allocation sites with source-level line references.
- Add a native benchmark that separately measures training, evaluation, prefill, decode, p50/p95 per-token latency, allocations, resident memory, and thread scaling.
- Cross-check the existing counters against repeated runs and CPU affinity.
- Research authoritative complexity and systems baselines for recurrent, state-space, and attention architectures.
- Convert findings into a prioritized upgrade plan with mandatory gates before any further language training.

## Realistic scaling estimates

The current formulas use separate input embeddings and output projection, and store parameters, first moments, and second moments as `double`. The following estimates exclude activations, datasets, allocator overhead, and runtime buffers:

| Active vocab | Embedding/hidden | CCT parameters | CCT parameter bytes | CCT Adam parameter+moments |
|---:|---:|---:|---:|---:|
| 32,768 | 512 | 34,637,312 | 277.1 MB | 831.3 MB |
| 50,257 | 768 | 79,606,609 | 636.9 MB | 1.91 GB |
| 100,000 | 1,024 | 209,097,376 | 1.67 GB | 5.02 GB |

At the current compact pilot dimensions (513 slots, width 16), this memory problem is hidden. At realistic language-model dimensions, the output head and untied embedding matrix dominate, while text-decimal checkpoints would be operationally unacceptable. Weight tying, lower-precision storage/optimizer state, sharded or fused vocabulary projection, and binary checkpoints are mandatory scale investigations before any broad training claim.
