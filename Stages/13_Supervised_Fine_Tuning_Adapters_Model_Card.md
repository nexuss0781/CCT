# Stage 13 Supervised Fine-Tuning and Adapters — Model Card

## Model identity

This card describes the bounded native C++20 Stage 13 task-adaptation pilot. It is a reproducibility and safety artifact, not evidence of a general instruction-following model.

| Field | Value |
|---|---|
| Base checkpoint | Stage 11 native checkpoint |
| Base checkpoint SHA-256 | `8ff1f227513d79a840b648bd724823e3fd790ba3bd9e754a086f430ebbd81b62` |
| Representation | Stage 10 immutable hybrid tokenizer snapshot |
| Task classes | Classification, extraction, grounded QA, summarization, code understanding, workflow drafting |
| Training/evaluation examples | 9 / 9 |
| Full SFT parameters | 18 |
| Rank-1 adapter parameters | 10 |
| Training authorization | `false` |

## Intended use

The implementation is intended for native supervised-adaptation research: validating deterministic instruction formatting and target-only masks, comparing full and parameter-efficient task updates, testing structured-output and citation boundaries, measuring base retention, and enforcing adapter/data permissions and deletion lineage.

It is not intended for production serving, high-impact decision-making, autonomous workflow execution, unsupervised external action, or replacing expert or human review.

## Data and provenance

The pilot uses bounded examples derived from declared Project Gutenberg and native CCT source fixtures already governed by earlier stages. Each example has independently hashed input and target content, an example hash, source and target provenance, task/schema identity, split, policy class, evaluator owner, and explicit training/evaluation permissions. The valid manifest contains nine training and nine held-out evaluation examples across six task classes. An evaluator-only contamination attempt is rejected rather than admitted to training.

The rights and privacy status of the underlying fixtures remain governed by Stage 9. This card does not establish universal licensing, privacy completeness, or representativeness.

## Adaptation methods

The full path updates all parameters of the declared small task head with clipped categorical gradients. The adapter path freezes the base and learns a rank-1 low-rank output-projection factor. Adapter metadata contains task, domain, version, rank, target module, base checkpoint hash, training-manifest hash, and permissions. The registry denies mismatched task, base, or permission requests. Merged and runtime predictions agree on the declared deterministic structured fixture.

## Evaluation evidence

The Stage 13 gate passed all eight mandatory checks. Full SFT improved the primary held-out classification fixture for seeds 3, 5, and 7. The selected bounded run improved accuracy from `0.0` to `1.0`; cross-entropy changed from `0.700682` to `0.0160871`, `0.0120894`, and `0.0140864` across the three seeds. Structured extraction reached held-out accuracy and schema validity of `1.0`. The adapter used 10 trainable parameters versus 18 for full tuning, while the base checksum remained unchanged.

Supported grounded answers retained the declared citation ID, missing-evidence questions abstained, malformed structured output was rejected, four unsafe action/secret requests were denied, and a deleted example was absent from the replacement manifest.

These are small fixture results. They do not establish broad task generalization or factuality.

## Compatibility and reproducibility

The Stage 13 executables are `cct_sft_tests` and `cct_stage13_gate`; the complete sequential command is `make ci-stage13`. The release artifacts include task and split manifests, formatter/mask report, task comparison, retention, adapter registry, merge parity, deletion, efficiency, review, incident, and release records under `artifacts/stage-13/cpp-gate/`.

## Risks and limitations

The model is tiny and the features are deterministic reference features rather than a production representation. Structured validity may coexist with incorrect content. Citation integrity is checked against declared fixture IDs rather than an open-world retrieval system. Adapter quality depends on the task fixture and the single tested output-projection placement. The review result is a bounded expert-proxy check and does not replace independent human review for high-impact domains. Retention is measured only on declared fixtures. No preference optimization, external action, production serving, or broad safety claim is included.

## Non-claims

Stage 13 does not claim general instruction following, factual reliability, broad multilingual or code competence, preference alignment, autonomous agency, high-impact safety, production readiness, or superintelligence. `training_authorized` remains false. Stage 14 requires a new specification, gate, and explicit approval.
