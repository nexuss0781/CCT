# Stage 11 Trainable Native NLP Core — Model Card

## Model identity

This model card describes the **Stage 11 CCT native next-token training pilot** in the CCT-ASE research repository. It is an engineering artifact for reproducibility and review, not a claim that the repository contains a general-purpose language model.

| Field | Value |
|---|---|
| Model family | CCT selective recurrent next-token model |
| Implementation | Native C++20 |
| Selected pilot seed | `3` |
| Embedding / hidden dimensions | `2 / 2` |
| Context length | `24` |
| Parameter count | `3,862` |
| Tokenizer | Stage 10 hybrid snapshot |
| Tokenizer SHA-256 | `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a` |
| Dataset SHA-256 | `d8c7c24937e7603064a1a5d3b07b0472fe672426b52d00e41b2b4d614c240996` |
| Selected checkpoint SHA-256 | `8ff1f227513d79a840b648bd724823e3fd790ba3bd9e754a086f430ebbd81b62` |
| Training authorization | `false` |

## Intended use

The implementation is intended for **native training-system research**: verifying categorical next-token loss, analytic recurrent gradients, deterministic optimizer updates, governed train/validation identity, checkpoint interruption and recovery, matched-control accounting, and bounded CPU resource measurement. It may be used as a reference implementation for later scaling experiments after a separate approval and gate.

It is not intended for production language serving, autonomous decision-making, high-stakes advice, safety-critical control, unrestricted generation, user profiling, or claims of general intelligence.

## Data and provenance

The Stage 11 gate uses bounded slices of declared real and application-shaped fixtures. The training side includes Project Gutenberg `pg1342.txt`, native CCT C++ source fixtures, code/JSON/Unicode/separator fixtures, and deterministic context windows. Validation includes Project Gutenberg `pg11.txt` and a held-out native-source slice. The gate records source-derived tokenizer and dataset hashes, split identities, window counts, active target counts, and explicit eligibility flags.

Training records require `training_allowed=true`. Validation records require `evaluation_allowed=true`. Evaluator-only records are marked `evaluator_only=true` and are rejected from both dataset splits. The pilot contains 37 training windows and 18 validation windows, with 782 active training targets and 389 active validation targets. The source rights and usage declarations originate from the Stage 9 governed corpus manifest; this model card does not replace legal review or establish universal licensing status.

## Architecture and training

The CCT candidate uses trainable sparse-domain token embeddings, selective retain/write gates, a nonlinear recurrent candidate, a previous-input projection, and an untied vocabulary head. It is trained with stable categorical cross-entropy over explicit loss masks, analytic backpropagation through the recurrence, global gradient clipping, warmup and linear decay, and AdamW-equivalent first/second moments. The checkpoint records model configuration, optimizer moments, schedule, tokenizer hash, dataset hash, data cursor, seed/state label, history, and model parameters.

The gate also runs native dense causal attention, GRU, and diagonal SSM controls with the same token-ID domain, context length, training fixture, optimizer budget, and CPU environment. The matched control parameter counts are 3,852, 3,870, and 3,846 respectively; the CCT count of 3,862 is within the declared ±10% band.

## Evaluation evidence

All eight Stage 11 mandatory gate checks passed. The selected seed-3 CCT pilot reduced held-out cross-entropy from `6.877456` to `5.922243`, a relative improvement of `13.889%`, with final perplexity `373.248`. The other two seeds improved by `15.167%` and `14.141%`. The repeated-corpus overfit fixture reduced training loss by `70.682%`. The analytic gradient spot check met the `1e-4` relative-error threshold. Checkpoint interruptions at cursors `0`, `1`, and `3` reproduced uninterrupted training within `1e-12` for the deterministic CPU pilot.

These results show that the implemented training objective can optimize the declared bounded fixture. They do not establish generalization beyond that fixture or superiority over production Transformer implementations.

## Compatibility and reproducibility

The trainer rejects incompatible tokenizer and dataset hashes, malformed or truncated checkpoints, invalid token IDs, non-finite parameters, all-false masks, evaluator-only records, and optimizer-state size mismatches. The immutable input snapshot is tracked at `data/stage-10/tokenizer_snapshot.bin`. The native regression executable is `cct_nlp_trainer_tests`; the artifact-producing gate is `cct_stage11_gate`; the complete sequential command is `make ci-stage11`.

## Risks and limitations

The model is intentionally tiny and trained on a very small declared pilot. The vocabulary is sparse-domain and the output head is untied. The native matched baselines are reference controls, not optimized industrial implementations. Throughput is measured on the declared sandbox CPU and must not be interpreted as deployment performance. The validation result may be sensitive to fixture composition, context windowing, seed, optimizer schedule, source-domain overlap, and the small data scale.

The checkpoint preserves the deterministic pilot seed/state label and data order, but this stage does not claim universal replay across different compilers, hardware, floating-point libraries, or parallel execution schedules. Privacy detection, rights review, factuality, safety behavior, multilingual coverage, long-context quality, distributed recovery, serving, human evaluation, and external safety review remain outside scope.

## Non-claims

Stage 11 does not claim broad language competence, factual knowledge, instruction following, grounded generation, safety alignment, production readiness, efficient large-scale training, Transformer replacement, autonomous agency, or superintelligence. The release record keeps `training_authorized=false`; later training, scaling, fine-tuning, alignment, retrieval, serving, and deployment require separate stage specifications, evidence, and explicit approval.
