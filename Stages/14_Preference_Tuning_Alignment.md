# Stage 14 — Preference Tuning and Alignment
## Human Preferences, Refusal Quality, and Behavior Regression Control

**Predecessor:** Stage 13 — Supervised Fine-Tuning and Adapters  
**Successor:** Stage 15 — Verified Retrieval and Knowledge Plane  
**Status:** Specification; implementation not started  
**Implementation:** Native C++20 preference-data and alignment pipeline

## Purpose

Stage 14 improves controllability, helpfulness, refusal quality, citation behavior, and output style using explicit preference evidence while protecting truthfulness, calibration, task quality, and safety. It compares preference optimization methods rather than assuming that one alignment technique is universally reliable.

## Scope and non-goals

The stage includes preference-rubric design, pairwise and scalar labels, rater training, disagreement analysis, supervised preference modeling, DPO-like optimization, reward-model plus policy optimization where justified, verifier-weighted reranking, adversarial preference data, refusal and uncertainty calibration, and broad regression evaluation. It does not authorize autonomous policy creation, unrestricted reward hacking, external actions, or production deployment.

## Preference data contract

Each preference item must include:

```text
preference_id
prompt_and_context
candidate_a
candidate_b
preferred_label
rater_or_judge_id_class
rubric_version
risk_category
conflict_or_tie_state
source_and_license
split_assignment
adjudication_state
```

Rater identity must be privacy-protected while preserving reviewer class, expertise, and conflict metadata. High-impact domains require domain-qualified reviewers. Disagreement and ties must remain visible rather than being collapsed into false certainty.

## Alignment candidates

| Candidate | Description | Required control |
|---|---|---|
| No preference tuning | SFT/base behavior | Quality and safety baseline |
| DPO-like | Direct preference classification-style objective | Compare data efficiency and stability |
| Reward-model + policy | Explicit reward model and policy optimization | Use only with separate reward validation |
| Verifier-weighted | Citation/schema/safety verifier signals | Test reward hacking and over-refusal |
| Reranking | Generate candidates and rank with approved verifier | Measure latency and diversity cost |

DPO is a candidate because its published formulation simplifies the RLHF procedure by deriving a preference objective without a separate reward-model and reinforcement-learning loop [1]. The roadmap does not treat that as proof that DPO is best for CCT; every method must face the same held-out and human evaluation.

## Required implementation

| Component | Implementation | Contract |
|---|---|---|
| Rubric registry | Versioned task and safety rubrics | Rubric changes produce new experiment identity |
| Preference loader | Manifested, split-safe pair loader | Evaluator-only data is inaccessible to training |
| SFT preference baseline | Preference-conditioned supervised model | Establishes non-optimization control |
| DPO-like trainer | Native objective and checkpointing | Exact data/temperature/reference model recorded |
| Reward path | Optional reward model and policy trainer | Reward model has held-out calibration |
| Verifier path | Citation/schema/policy/uncertainty scoring | Verifier cannot silently alter policy |
| Reranker | Candidate generation and scoring | Latency, diversity, and failure are measured |
| Adversarial data | Conflict, jailbreak, injection, reward-hacking fixtures | Unsafe preferences cannot become policy |
| Human review | Blind comparison and expert escalation | Review result is a release artifact |

## Evaluation harness

The harness must test:

1. preference-data schema and split integrity;
2. rater disagreement, ties, and adjudication;
3. held-out preference accuracy;
4. task quality and helpfulness;
5. factuality, grounding, and citation correctness;
6. refusal precision, refusal recall, and false-refusal rate;
7. calibration and abstention quality;
8. robustness to prompt injection and misleading context;
9. reward/verifier hacking attempts;
10. diversity, verbosity, and mode-collapse diagnostics;
11. catastrophic forgetting against the base and SFT models;
12. human blind comparisons and domain-expert review;
13. training stability, compute, memory, and inference overhead;
14. reproducibility across seeds and checkpoint resume.

A preference-tuned model must not be accepted because a single automated judge score rises. The harness must report quality and safety jointly, including cases where the correct behavior is uncertainty or refusal.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Preference integrity | Labels, rubrics, rater classes, conflicts, and splits are auditable |
| Held-out preference | Candidate improves over the no-preference control on held-out comparisons |
| Task retention | SFT task quality and schema validity remain within regression budget |
| Truthfulness | Unsupported-claim rate does not increase beyond threshold |
| Grounding | Citation precision and evidence adherence remain green |
| Refusal quality | Unsafe-request handling improves without unacceptable false refusals |
| Calibration | Confidence and abstention remain calibrated |
| Reward robustness | Injected reward/verifier hacks do not produce unsafe acceptance |
| Human review | Blind and domain-expert reviews pass declared rubrics |
| Stability | Training and resume are deterministic/tolerance-bounded |
| Efficiency | Added training and inference cost is measured and approved |
| Regression | Prior Stage 0–13 gates remain green |

## Pass/fail transition

Stage 14 passes only when a selected alignment method improves declared behavior without unacceptable regressions in truthfulness, safety, calibration, task quality, or operations. A `PASS` authorizes Stage 15 verified retrieval and knowledge-plane integration.

A `FAIL` requires method, data, rubric, or verifier remediation. A `BLOCKED` result is valid when preference provenance, rater safety, or domain expertise is unresolved.

## Deliverables

The stage must deliver preference/rubric schemas, rater protocol, preference dataset manifest, DPO-like and comparator trainers, verifier/reranker integration, adversarial suite, human-review report, alignment model card, native tests, gate executable, and CI command.

## Explicit limitations

Human preferences are not a complete definition of truth, safety, or usefulness. Automated judges can be biased or gamed. Preference tuning may increase stylistic agreement without increasing factual accuracy. High-impact domains require human and domain-specific review.

## References

[1]: https://arxiv.org/abs/2305.18290 "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
