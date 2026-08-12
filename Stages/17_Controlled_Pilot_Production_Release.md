# Stage 17 — Controlled Pilot and Production Release
## Shadow Evaluation, Human Oversight, and Bounded Deployment

**Predecessor:** Stage 16 — Production Inference and Operations  
**Successor:** New specification required for any expanded capability or external agency  
**Status:** Implemented; PASS — terminal bounded-release gate

**Implementation:** Native C++ release controls plus approved infrastructure integrations

## Purpose

Stage 17 is the final production-release gate for a narrowly defined CCT-NLP use case. It moves the validated model and service through shadow operation, limited pilot, human oversight, incident response, rollback, and release review. It is not a general autonomy gate and does not authorize unrestricted deployment.

## Scope and non-goals

The stage includes release candidate freeze, shadow traffic, offline/online parity, limited approved users, human review, user feedback, SLO measurement, safety monitoring, incident management, rollback, deletion requests, model/version changes, post-release evaluation, and final release documentation. It does not authorize high-consequence decisions, autonomous external actions, unrestricted tools, online self-improvement, or expansion to unapproved domains.

## Release boundary

The release record must explicitly name:

```text
approved_model_version
approved_tokenizer_version
approved_adapter_versions
approved_retrieval_index_version
approved_task_schemas
approved_user_groups
approved_tenant_boundaries
approved_data_classes
approved_regions/jurisdictions
approved_tool_permissions
human_approval_requirements
service_level_objectives
rollback_version
release_expiration_or_review_date
```

Anything not listed is denied by default. A model cannot inherit broader permissions merely because its predecessor had them.

## Release phases

| Phase | Description | User-visible effect |
|---|---|---|
| R0 artifact freeze | Freeze model, tokenizer, adapters, index, policy, service, and manifests | None |
| R1 offline replay | Replay locked evaluation and representative traces | None |
| R2 shadow | Process mirrored requests without returning outputs or actions | None |
| R3 internal pilot | Approved staff/users review outputs in low-risk workflows | Controlled internal access |
| R4 limited pilot | Small approved user group with quotas and human escalation | Limited user-visible output |
| R5 production | Declared low-risk use cases under SLO and incident monitoring | Bounded production |

Each phase requires a written decision. Passing R2 does not automatically authorize R3; passing R4 does not automatically authorize expansion.

## Human oversight

Human review must be designed into the workflow rather than added after an incident. Reviewers must see the model output, uncertainty, citations, policy decision, relevant trace, and escalation options. High-impact or ambiguous outputs must be routed to review or abstention. User feedback must distinguish quality complaints, factual errors, privacy issues, unsafe content, policy failures, and infrastructure failures.

## Required implementation

| Component | Implementation | Contract |
|---|---|---|
| Release manifest | Immutable model/service/data/policy identity | Every request resolves to a release |
| Shadow runner | Mirrored request replay without side effects | Outputs are isolated and auditable |
| Pilot controller | User/tenant allowlist, quotas, expiration | Unauthorized use is denied |
| Review console/API | Output, citations, uncertainty, trace, escalation | Human decision is logged |
| SLO monitor | Quality, safety, latency, cost, availability | Alerts have owners and thresholds |
| Incident manager | Severity, containment, investigation, remediation | No silent closure |
| Feedback store | Structured, privacy-controlled feedback | Feedback cannot directly change weights |
| Rollback | Atomic prior-version restoration | Rollback is tested before release |
| Deletion handler | Data/user/model artifact deletion workflow | Deletion is auditable |
| Change control | Model/index/policy/config approval | Every change triggers relevant regression |
| Post-release evaluator | Locked and fresh evaluation sets | Drift and regression are measured |

## Shadow and pilot evaluation

Shadow traffic must compare the release candidate with the current control on identical requests where permitted. It must measure:

1. output quality and task success;
2. citation correctness and unsupported-claim rate;
3. abstention and refusal quality;
4. calibration and human preference;
5. latency and resource cost;
6. retrieval/index behavior;
7. tenant and policy isolation;
8. incident and error rates;
9. disagreement between CCT-only, hybrid, and Transformer controls;
10. rollback and recovery readiness.

The pilot must include negative controls, adversarial traffic, malformed input, missing data, stale data, conflicting evidence, quota abuse, timeout, dependency failure, and policy-denied actions. It must not use user data for further training by default.

## Evaluation harness

The harness must execute locked offline replay, shadow traffic comparison, controlled pilot sampling, human-review workflows, SLO measurement, safety/adversarial traffic, deletion requests, rollback rehearsal, incident drills, and drift detection. It must preserve evaluator-only labels and report quality, safety, latency, cost, availability, human escalation, and rollback outcomes by approved use case and user group.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Artifact freeze | All release identities and hashes are immutable and complete |
| Offline parity | Release reproduces locked evaluation artifacts |
| Shadow | Candidate is measured against control without side effects |
| Quality | Declared task quality and grounded-citation thresholds pass |
| Safety | Adversarial, privacy, prompt-injection, and policy suites pass |
| Human oversight | Required reviews and escalation paths function in rehearsal |
| SLO | Declared latency, availability, throughput, and cost targets pass |
| Isolation | Tenant, data, adapter, state, and permission boundaries hold |
| Rollback | Prior release restores within declared target under fault injection |
| Incident response | Alerts, ownership, containment, and postmortem path are tested |
| Deletion | Approved deletion requests propagate to service and derived artifacts |
| Drift | Monitoring detects quality, safety, data, and workload drift |
| Regression | Prior Stage 0–16 gates remain green |
| Approval | Named technical, security, product, and governance approvers sign release |

## Release decision

The final decision must be one of:

| Decision | Meaning |
|---|---|
| `PASS — bounded production` | Approved only for the declared scope and time window |
| `PASS — limited pilot` | Evidence is promising but production SLO or coverage is incomplete |
| `HOLD` | Remediation, more data, or more human review is required |
| `FAIL` | Release candidate is not acceptable |
| `BLOCKED` | Governance, security, infrastructure, or ownership is unresolved |

A `PASS — bounded production` must include an expiration or review date. New domains, new data classes, new tools, larger user populations, autonomous action, or online learning require a new release review.

## Incident and rollback policy

Any secret exposure, tenant crossover, unreviewed external action, policy bypass, material privacy event, persistent unsupported high-impact output, unrecoverable audit gap, or failed rollback is an automatic release stop. The service must enter a safe degraded mode or revert to the last valid release. The incident record must preserve evidence, timeline, scope, containment, root-cause hypothesis, remediation, and approval to resume.

## Deliverables

The stage must deliver the release manifest, shadow and pilot reports, SLO dashboard/export, human-review protocol, feedback and incident schemas, rollback rehearsal, deletion rehearsal, drift report, model/system card, final approval record, native regression and release gates, and a bounded production runbook.

## Pass/fail transition

Stage 17 passes only when the exact release scope is approved, shadow and pilot evidence is complete, quality and safety thresholds pass, SLOs are met, human oversight and rollback work, and all incidents are accounted for. The result authorizes only the named bounded production use case.

There is no automatic Stage 18. Any future work involving broader autonomy, external tools, persistent identity, online learning, high-consequence domains, real-world embodiment, or autonomous research must begin with a new specification and threat model.

## Implemented gate scope

The native implementation provides immutable release-scope identity, model/tokenizer/adapter/index binding, sequential R0–R5 phase decisions, locked offline replay, shadow comparison without side effects, approved user and tenant allowlists, quotas and expiration, human review and escalation, structured feedback that cannot update weights, quality/safety/latency/availability/cost observations, incident containment and resume approval, deletion propagation to service state/cache/derived artifacts, drift detection and ownership, rollback rehearsal, named technical/security/product/governance approval signatures, terminal release evaluation, and bounded runbook/model-card artifacts.

The terminal gate is wired into `ci-stage17` after all Stage 0–16 checks. Its release record authorizes only the named low-risk task and scope. External actions, unrestricted tools, online learning, high-consequence decisions, and automatic expansion remain denied. Any future expansion requires a new specification and threat model.

## Explicit limitations

A limited pilot cannot prove safety or quality for all users, languages, domains, or future model versions. Production monitoring is necessary but does not eliminate risk. Human review can fail and must itself be evaluated. Release authority remains a governance decision, not a consequence of benchmark performance alone.
