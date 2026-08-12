# Stage 8 Expanded Gate Contract — Production Foundation and Governance

**Stage:** 8  
**Implementation:** Native C++20 only  
**Scope:** Production foundation, governance, realistic application readiness, evaluation registry, and release controls  
**Transition:** Stage 9 implementation only after explicit approval

## Gate intent

This gate validates that the production NLP program is implementable and governable before any large training or deployment work begins. It is not a language-model capability gate. It must prove that the repository contains executable governance contracts, realistic application fixtures, adversarial controls, artifact schemas, decision records, and code-quality evidence.

## Required native components

| Component | Required result |
|---|---|
| Product boundary | Versioned use-case registry with allowed and disallowed actions |
| Threat model | Versioned threats, controls, owners, tests, and residual risk |
| Data policy | Source/license/privacy/retention classes and fail-closed decisions |
| Experiment registry | Immutable experiment identity, seed, config, data, hardware, and status |
| Evaluation registry | Held-out tasks, baselines, metrics, negative controls, and owners |
| Artifact registry | Hashable manifest for code, data, config, binaries, and reports |
| Release policy | Stage status, approver, scope, expiration, rollback, and no-claim boundary |
| Application fixtures | Structured extraction, grounded answer, classification, and code-understanding workflows |
| Policy engine | Deny-by-default external action, online learning, secret, and host-execution decisions |
| Audit | Complete policy, experiment, evaluation, and release trace |

## Realistic application fixture contract

The gate uses deterministic but application-shaped fixtures, not toy arithmetic-only tests. Each fixture contains a user request, tenant, source documents, task schema, expected structured output or label, evidence requirements, policy state, allowed action set, and evaluator-only truth. Fixtures include:

1. invoice or purchase-order field extraction with schema validation;
2. support-ticket classification and priority routing without executing the route;
3. internal-policy question answering requiring citations and abstention on missing evidence;
4. source-code change explanation using static analysis only;
5. conflicting-document resolution requiring uncertainty;
6. unauthorized external-action request requiring policy denial;
7. prompt-injection text embedded in a source document;
8. malformed or privacy-sensitive input requiring quarantine or refusal.

No fixture is accepted as production evidence unless the application contract, evaluator truth, access policy, and expected failure behavior are recorded separately.

## Mandatory checks and thresholds

| Check | Pass condition |
|---|---|
| Stage inventory | Stage 8–17 specifications exist and each has required sections |
| Dependency integrity | No stage bypasses predecessor approval; dependencies are acyclic |
| Product boundary | 100% of fixtures map to a declared use case and action scope |
| Data governance | Training/evaluation/private/source classes are distinct and fail closed when unresolved |
| Threat coverage | Every high-severity threat has a control, test, owner, and residual-risk state |
| Policy safety | External action, host execution, secret access, online learning, and autonomous self-modification deny by default |
| Application readiness | All eight application-shaped fixtures produce the expected policy/evidence decision path |
| Negative controls | Injected missing, conflicting, unauthorized, and poisoned inputs are rejected or abstained from |
| Artifact integrity | Manifest hashes and config/commit identity reproduce exactly |
| Evaluation registry | Every future capability claim names task, split, baseline, metric, seed, and evaluator |
| Code quality | Strict C++20 build passes with warnings as errors; no TODO/FIXME/debug leakage in release paths |
| Auditability | All governance decisions and fixture outcomes are reconstructable |
| Reproducibility | Two same-seed runs produce identical governance/application artifacts |
| Claim boundary | Reports separate implemented behavior, proposal, hypothesis, and non-claim |

## Required artifacts

The gate must write:

```text
artifacts/stage-8/cpp-gate/
├── gate.json
├── checks.json
├── metrics.json
├── manifest.json
├── product_registry.json
├── threat_model.json
├── evaluation_registry.json
├── application_visible.json
├── evaluator_truth.json
├── audit.jsonl
├── incident_log.json
└── report.md
```

Evaluator-only truth must not be copied into the public report. The artifact manifest must identify publishable versus restricted files.

## Automatic failure conditions

The gate fails automatically for any unauthorized action marked allowed, any unresolved high-severity threat represented as accepted without an owner, any application fixture whose evidence or policy outcome is silently missing, any evaluator contamination, any artifact hash mismatch, any strict-build warning, any hidden exception to deny-by-default policy, any unlogged governance decision, or any claim that calls the production NLP engine implemented when it is only specified.

## Transition decision

`PASS` authorizes Stage 9 implementation only. It does not authorize training, public data acquisition, model deployment, external tools, online learning, or a production claim. `FAIL` requires remediation and rerun. `BLOCKED` is acceptable for unresolved legal, privacy, security, ownership, or infrastructure decisions, but blocked items cannot be presented as passed readiness evidence.
