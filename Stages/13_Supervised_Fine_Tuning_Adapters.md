# Stage 13 — Supervised Fine-Tuning and Adapters
## Task Adaptation, Structured Outputs, and Capability Retention

**Predecessor:** Stage 12 — Scaling and Accelerator Systems
**Successor:** Stage 14 — Preference Tuning and Alignment
**Status:** Implemented and gated
**Implementation:** Native C++20 supervised fine-tuning, low-rank output adapters, formatter/masks, registry authorization, structured validation, retention, and gate

## Purpose

Stage 13 turns the validated native next-token core into a bounded task-adaptation system. It tests deterministic instruction formatting, explicit loss masks, full-parameter supervised fine-tuning, parameter-efficient low-rank adaptation, structured-output validation, grounded citation behavior, safety retention, adapter authorization, deletion lineage, and reproducible artifact identity.

This is a small controlled CPU adaptation pilot, not a claim of broad instruction following, factuality, human-level review, production deployment, or general intelligence.

## Scope and non-goals

The implemented scope covers six task classes, independent instruction-example hashes, task/schema/policy registries, train/evaluation eligibility, deterministic formatter output, target-span-only masks, full SFT updates, low-rank output-projection adapters, adapter permissions, base immutability, merged/runtime parity, structured JSON validation, grounded citation and missing-evidence abstention, workflow approval metadata, deletion/rebuild lineage, retention/safety fixtures, and native artifact production.

The stage does not include preference optimization, external actions, host execution, high-impact deployment, human-review replacement, retrieval infrastructure, production serving, or final release.

## Implemented scope

The public API is in `cpp/include/cct/sft.hpp` and the implementation is in `cpp/src/sft.cpp`. `SftManifest` validates task IDs, schema versions, independent source/target/example hashes, split permissions, evaluator ownership, and evaluator exclusion. `SftFormatter` serializes a versioned task envelope and masks only target-span tokens. `SftModel` provides deterministic task features, categorical loss, analytic full-update gradients, structured predictions, and serialization. `SftAdapter` freezes the base model and trains low-rank factors on the output projection. `SftAdapterRegistry` enforces task, base-checkpoint, and permission matching. `StructuredDecoder` rejects overlong or malformed outputs and forces abstention where the task permits it.

The Stage 13 gate is `cpp/tools/stage13_gate.cpp`; the regression executable is `cct_sft_tests`. The immutable Stage 11 base checkpoint identity is retained as `8ff1f227513d79a840b648bd724823e3fd790ba3bd9e754a086f430ebbd81b62`. The Stage 10 tokenizer snapshot remains the representation identity.

## Supported task registry

The pilot registers all six required classes.

| Task | Output contract | Safety/control behavior |
|---|---|---|
| Classification | Closed label plus confidence | Unknown labels fail closed |
| Structured extraction | Bounded JSON object with spans | Malformed JSON is rejected |
| Grounded question answering | Answer, citation, uncertainty, abstention | Missing evidence abstains |
| Summarization | Bounded summary output | Unsupported claims remain outside scope |
| Code understanding | Structured issue/clean result | No host execution |
| Workflow drafting | Draft plus approval-required metadata | Submission is not implemented |

## Data and formatting contract

The gate uses nine governed training examples and nine held-out evaluation examples derived from declared Project Gutenberg and native CCT source fixtures. The examples contain task IDs, schema versions, source/target provenance, policy classes, split assignments, evaluator ownership, independent hashes, and explicit eligibility. The evaluator-only contamination path is tested separately and rejected.

The formatter emits a deterministic `<CCT_TASK_V1>` envelope containing task, schema, input, target marker, target, and end marker. Only target-span content tokens receive loss. Header, input, control, and boundary positions are masked. Reformatting the same example with the same tokenizer is byte- and mask-identical.

## Adaptation paths

Full SFT updates all 18 parameters of the declared two-label task head with clipped categorical gradients. The adapter path freezes the 18-parameter base vector and trains a rank-1 output-projection factor with 10 trainable parameters. The adapter records task/domain/version, rank, target module, base checkpoint hash, training-manifest hash, and permissions.

The gate compares untouched base, full SFT, adapter SFT, and a no-training control. Three seeds `3`, `5`, and `7` are used for the primary classification task. The selected adapter remains separate from the base and is merged only through an identity-checked operation.

## Mandatory gate checks

All eight Stage 13 gate checks passed.

| Check | Result | Evidence |
|---|---|---|
| Task registry and provenance | **PASS** | Six task classes; 18 examples; 9 train and 9 held-out evaluation; evaluator training 0 |
| Formatter and masks | **PASS** | `target-span-only-v1`; deterministic active target mask |
| Three-seed full SFT | **PASS** | Seeds 3, 5, and 7 each improved held-out classification |
| Representative task improvement | **PASS** | Classification and structured extraction improved; schema output valid |
| Adapter efficiency/isolation | **PASS** | 10 adapter parameters versus 18 full parameters; base immutable; unauthorized loads denied |
| Citation/safety retention | **PASS** | Supported citation valid; missing evidence abstains; four unsafe requests denied |
| Deletion/fail-closed inputs | **PASS** | Deleted example absent from replacement manifest; evaluator record rejected |
| Artifact/review identity | **PASS** | Base checkpoint and manifest identity recorded; bounded expert-proxy review passed |

The three-seed classification results were finite and improved from base accuracy `0.0` and cross-entropy `0.700682` to adapted accuracy `1.0` with cross-entropies `0.0160871`, `0.0120894`, and `0.0140864` for seeds 3, 5, and 7. Structured extraction reached accuracy and schema validity `1.0` on its held-out fixture. These are bounded fixture results, not broad task claims.

## Evaluation harness

The regression suite covers manifest replay, deterministic formatter/masks, full SFT gradients and learning, adapter freeze/gradient/registry behavior, structured merge parity, malformed-output rejection, evaluator-only exclusion, and finite metrics. The artifact-producing gate covers all six task classes, three primary seeds, base/full/adapter/no-training comparison, citation integrity, missing-evidence abstention, safety retention, deletion lineage, and expert-proxy review.

The harness includes conflicting or missing evidence, malformed structured output, unknown or unsafe workflow requests, code fixtures, Unicode-capable tokenizer inheritance, and unauthorized adapter identity/permission combinations. The model never performs external actions.

## Deliverables

The stage delivers the native SFT API and implementation, expanded gate contract, six-task registry, formatter and loss masker, full and adapter paths, adapter registry, structured decoder, regression suite, gate executable, CMake/CTest integration, Makefile targets, task comparison, retention report, deletion report, merge-parity report, efficiency report, review report, incident log, release record, and human-readable report under `artifacts/stage-13/cpp-gate/`.

The canonical commands are:

```bash
make stage13-test
make stage13-gate
make ci-stage13
```

## Pass/fail transition

Stage 13 passes because two representative tasks improve over the untouched base, structured output validation succeeds, grounded citation and missing-evidence behavior remain bounded, unsafe requests remain denied, adapter permissions and base immutability hold, merged/runtime outputs agree, deleted data is absent from the rebuilt manifest, and all artifacts are identity-linked.

A Stage 13 `PASS` authorizes Stage 14 preference-tuning engineering within its own gate. It does not authorize preference optimization, external actions, high-impact use, or production release.

## Explicit limitations

The pilot is intentionally tiny, CPU-bound, and based on declared fixtures. The task features and output head are reference adaptation paths, not a production instruction model. Structured validity does not prove factuality. Citation checks use declared fixture IDs rather than open-world retrieval. The expert-review result is a bounded proxy and does not replace independent human review for high-impact tasks. Retention evidence is limited to the declared fixtures. Adapter placement is tested only on the output projection; input, recurrent, memory, and verifier placements remain future ablations. `training_authorized` remains false.

## References

[1]: https://arxiv.org/abs/2106.09685 "LoRA: Low-Rank Adaptation of Large Language Models"
