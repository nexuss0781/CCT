# Stage L1-8 — Operation and API Teacher Adaptation

**Predecessor:** L1-7 — Supervised Instruction Adaptation  
**Successor:** L1-9 — Bounded Teaching Behavior  
**Implementation language:** Native C++20  
**Status:** Implementation contract; transition remains approval-gated until the formal gate is published from the release commit.

## Objective

Teach CCT to represent, serialize, explain, validate, correct, and safely reject declared internal operation and API calls. The stage measures a bounded operation-contract teacher interface; it does not claim broad API competence, production deployment, unrestricted tool use, autonomous execution, or general intelligence.

## Required native components

| Component | Contract | Failure boundary |
|---|---|---|
| Versioned schema registry | Every operation has an immutable ID, schema version, description, authorization class, typed fields, required/optional status, bounds, defaults, and side-effect declaration. | Duplicate, empty, unsupported-version, duplicate-field, invalid-bound, non-finite, or side-effectful schemas are rejected. |
| Typed operation call | Calls carry request, tenant, user, role, operation, schema/manifest/checkpoint identities, typed arguments, evidence, ambiguity, and external-action flags. | Missing identity, unsupported version, unknown field, duplicate argument, wrong type, malformed value, trailing data, or identity mismatch is rejected. |
| Authorization | `public_read`, `tenant_member`, `reviewer`, and `admin` classes are evaluated against authenticated tenant/user/role/operation context. | Unauthenticated, cross-tenant, role-inadequate, or allow-list-inadequate calls are denied. |
| Validator and normalizer | Validation occurs before acceptance; optional defaults are inserted deterministically and normalized calls are serialized canonically. | Required-field, enum, numeric-bound, byte-bound, evidence, and ambiguity violations fail closed or abstain where declared. |
| Demonstration manifest | Demonstrations preserve source ID/span/hash, split, evaluator status, expected decision/error, correction, call lineage, and canonical demonstration identity. | Duplicate demonstrations, stale identities, evaluator-only training, missing provenance, and changed manifests invalidate the teacher. |
| Checkpoint identity | Operation schema registry, demonstration manifest, base checkpoint, tokenizer, and training configuration hashes are bound into one identity. | Any schema, data, model, tokenizer, or configuration change invalidates incompatible calls and teacher state. |
| Side-effect boundary | The L1 teacher returns validation, explanation, correction, and audit data only. | External actions, host execution, secret access, online learning, and self-modification are never performed. |

## Declared operation slices

The frozen gate fixture contains three representative schemas: `document.summarize` for tenant-member access and optional defaults, `knowledge.lookup` for reviewer-only evidence-bound access, and `workflow.draft` for admin-only reviewable drafting. The fixture is intentionally small but exercises distinct field types, authorization classes, evidence requirements, and safety boundaries.

## Failure and error contract

The response must expose a deterministic decision, error class, stable error code, explanation, correction, serialized normalized call where available, audit digest, and an explicit `side_effect_performed=false` field. The mandatory error classes are schema-version mismatch, identity missing, unknown operation, required field missing, unknown field, type mismatch, bounds violation, enum violation, authorization denied, ambiguous request, evidence missing, side-effect denied, duplicate argument, serialization error, and identity mismatch.

Ambiguous requests and missing required evidence produce `ABSTAINED` decisions. Invalid, unknown, unauthorized, malformed, side-effectful, or identity-incompatible calls produce `REJECTED` decisions. No error path may silently coerce an undeclared field or execute a side effect.

## Gate requirements

The independent native gate must include all of the following check families:

1. complete versioned schema registry and identity hash;
2. required and optional field declarations, typed bounds, defaults, and deterministic normalization;
3. valid call serialization and schema validation;
4. representative error-class mapping for missing, wrong-type, out-of-bounds, and unknown fields;
5. unknown-operation rejection;
6. authorization-class and operation allow-list enforcement;
7. ambiguity and evidence-boundary behavior;
8. explanation and correction linkage to the declared schema;
9. demonstration provenance and train/evaluator split isolation;
10. schema, manifest, checkpoint, tokenizer, and configuration identity binding;
11. teacher and call serialization replay equivalence;
12. stale schema, manifest, and checkpoint identity rejection;
13. explicit side-effect isolation;
14. strict trailing-data and corruption rejection;
15. schema evolution invalidation of old identities;
16. deterministic repeated replay; and
17. unauthenticated and unknown-operation negative controls.

A stage gate passes only when every mandatory family is `PASS`, the focused unit suite is green, the full native suite remains green, and every artifact is linked to the release commit and frozen identity values.

## Artifacts

The gate publishes `checks.json`, `operation_schema_registry.json`, `operation_manifest.json`, `formatter_validator_report.json`, `error_class_report.json`, `authorization_report.json`, `checkpoint_identity_report.json`, `side_effect_isolation_report.json`, `release_record.json`, and `report.md` under `artifacts/l1-8/cpp-gate/`. The release record keeps `training_authorized=false` and `approval_required=true`.

## Transition

`PASS` authorizes preparation of L1-9 bounded teaching evaluation only after explicit user approval. It does not authorize external operations, production API exposure, autonomous action, online learning, or unrestricted tool execution.
