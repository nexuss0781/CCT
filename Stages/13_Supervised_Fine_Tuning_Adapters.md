# Stage 13 — Supervised Fine-Tuning and Adapters
## Task Adaptation, Structured Outputs, and Capability Retention

**Predecessor:** Stage 12 — Scaling and Accelerator Systems  
**Successor:** Stage 14 — Preference Tuning and Alignment  
**Status:** Specification; implementation not started  
**Implementation:** Native C++20 fine-tuning and adapter pipeline

## Purpose

Stage 13 turns the validated base language model into a useful task model without losing general capability, safety behavior, provenance, or operational reproducibility. It implements supervised fine-tuning (SFT), full-model adaptation for small models, parameter-efficient adapters for larger models, structured output constraints, and task-specific evaluation.

## Scope and non-goals

The stage includes instruction formatting, task schemas, loss masking, SFT data manifests, full fine-tuning, low-rank or equivalent CCT adapters, adapter routing and isolation, structured decoding, checkpoint merge, task evaluation, capability-retention tests, and catastrophic-forgetting diagnostics. It does not include preference optimization, external actions, or final production release.

## Supported SFT task contracts

The first task registry must include at least one task from each applicable class:

| Task | Output contract |
|---|---|
| Classification | Closed label set plus calibrated confidence |
| Structured extraction | Versioned JSON/schema object plus source spans |
| Grounded question answering | Answer, citations, uncertainty, abstention |
| Summarization | Bounded length, factuality checks, source links |
| Code understanding | Explanation or structured issue report; no host execution |
| Workflow drafting | Draft only, with required human approval metadata |

Each item must include task ID, schema version, input provenance, target provenance, policy class, split, and evaluator ownership. Training and evaluation examples must be independently hashed.

## Required implementation

| Component | Implementation | Contract |
|---|---|---|
| Formatter | Conversation/task/schema serialization | Same item/version produces same tokens |
| Loss masker | Instruction, answer, citation, and control masks | Mask policy is explicit and tested |
| Full fine-tuner | All-parameter update for approved model sizes | Optimizer/checkpoint semantics match Stage 11 |
| Adapter tuner | LoRA-style or CCT low-rank adapters | Base weights remain immutable |
| Adapter registry | Task/domain/version/permissions | Unauthorized adapter load is denied |
| Structured decoder | JSON/schema/constrained output path | Invalid output is rejected or repaired safely |
| Merge tool | Base+adapter export | Merged and runtime outputs agree within tolerance |
| Retention evaluator | Base capability and safety regression suite | Release blocked on unacceptable regression |
| Data deletion | Adapter/data lineage and rebuild record | Deleted data does not remain silently in new artifact |
| Artifact card | Model/adapter/data/config and intended use | Every release is explainable |

## Adapter design

The adapter API must support attachment to the CCT input projection, recurrent update parameters, output projection, memory connector, verifier connector, or other explicitly approved modules. An adapter must declare trainable parameter count, rank or equivalent capacity, target modules, base-checkpoint hash, training data manifest, and permissions.

LoRA-style parameter-efficient adaptation is a candidate because it freezes base weights and adds trainable low-rank matrices, reducing downstream trainable parameters and memory requirements in the original study [1]. CCT must not assume that Transformer adapter placement transfers unchanged; each placement requires an ablation.

## Evaluation harness

The harness must compare base, full SFT, adapter SFT, and no-training controls on:

1. task quality and exact/schema validity;
2. citation and evidence correctness;
3. calibration and abstention;
4. general language and code retention;
5. long-context and Unicode robustness;
6. safety and refusal retention;
7. data deletion and artifact lineage;
8. adapter isolation and unauthorized-load rejection;
9. merged/runtime adapter parity;
10. parameter count, training memory, time, and serving overhead;
11. at least three seeds for the primary task;
12. human or expert review for representative outputs.

The harness must include deliberately conflicting, malformed, missing, and out-of-domain inputs. A model should abstain or produce a structured error rather than inventing a valid-looking answer.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Task improvement | SFT beats the base model on each declared target task |
| Schema validity | Structured-output validity meets the task threshold |
| Citation integrity | Grounded outputs cite correct evidence or abstain |
| Calibration | Confidence/selective-risk curve meets declared threshold |
| Retention | General, code, long-context, and safety regressions stay within budget |
| Adapter isolation | Unauthorized adapter/data access is denied |
| Merge parity | Merged and runtime adapters agree within tolerance |
| Data lineage | Model points to exact SFT data/config/checkpoint hashes |
| Deletion | Deletion workflow produces an auditable replacement artifact |
| Efficiency | Adapter training and serving costs are measured against full tuning |
| Human review | Expert rubric passes for declared use cases |
| Regression | Prior Stage 0–12 gates remain green |

## Pass/fail transition

Stage 13 passes only when at least two representative tasks improve, capability and safety retention are within declared bounds, adapter permissions are enforced, and artifacts are reproducible. A `PASS` authorizes Stage 14 preference tuning.

A `FAIL` requires data, formatting, masking, adapter, or model remediation. A `BLOCKED` result is valid when human review, task ownership, or data permissions are unresolved.

## Deliverables

The stage must deliver SFT data/task schemas, formatter and masker, full and parameter-efficient trainers, adapter registry, structured decoder, merge/parity tools, task reports, retention report, human-review report, native regression suite, gate executable, and CI command.

## Explicit limitations

SFT can improve a narrow task while harming other behavior. Adapter quality depends on task data and placement. Structured validity does not prove factual correctness. Human review remains necessary for high-impact domains.

## References

[1]: https://arxiv.org/abs/2106.09685 "LoRA: Low-Rank Adaptation of Large Language Models"
