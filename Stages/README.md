# CCT-ASE Stage Specifications

This directory contains the independent implementation and evaluation specifications for the CCT-ASE roadmap. Each stage defines its own scope, implementation contracts, evaluation harness, pass/fail criteria, transition package, and exit report. A later stage must not be treated as complete merely because its code exists; it requires a recorded `PASS` gate from the preceding stage.

## Stage map

| Stage | Document | Primary outcome | Transition |
|---|---|---|---|
| 0 | [Reproducible Baseline](00_Reproducible_Baseline.md) | Clean build, deterministic tests, CI, benchmark schema | Authorizes numerical work |
| 1 | [Differentiable Numerical Engine](01_Numerical_Engine.md) | Correct and differentiable field/operator substrate | Authorizes sequence core |
| 2 | [Efficient Sequence Core](02_Sequence_Core.md) | Stable selective recurrent model with streaming equivalence | Authorizes causal event learning |
| 3 | [Causal Event Learning](03_Causal_Event_Learning.md) and [Expanded Gate Contract](03_Expanded_Gate_Contract.md) | Native event identity, DAG structure, leakage audit, intervention, counterfactual, robustness, and abstention tests | PASS complete; Stage 4 requires explicit approval |
| 4 | [Persistent Verifiable Memory](04_Persistent_Verifiable_Memory.md) and [Expanded Gate Contract](04_Expanded_Gate_Contract.md) | Native checksummed log, exact retrieval, provenance/citations, deletion, conflicts, retention, poisoning boundary, and recovery | Stage 4 gate in progress; Stage 5 requires explicit approval |
| 5 | [Language and Code Scaling](05_Language_Code_Scaling.md) and [Expanded Gate Contract](05_Expanded_Gate_Contract.md) | Native small-scale language/code micro-run, matched baselines, provenance, checkpoint replay, memory attribution, long-context, and code-safety checks | Stage 5 gate in progress; Stage 6 requires explicit approval |
| 6 | [Deliberation and Verification](06_Deliberation_Verification.md) | Bounded planning, independent verification, and offline tools | Authorizes multimodal research |
| 7 | [Multimodal and Open-Ended Research](07_Multimodal_Open_Ended.md) | Controlled cross-modal transfer and research continuation decision | Requires research review |

## Global gate protocol

Every stage must produce an immutable or reviewable gate record containing the stage ID, repository commit, configuration hash, data or environment manifest hash, hardware, software versions, random seeds, test results, benchmark results, threshold definitions, known failures, and the final status.

The only valid statuses are:

| Status | Meaning | Allowed action |
|---|---|---|
| `PASS` | All mandatory criteria passed and transition artifacts are complete | Begin the successor stage within the documented scope |
| `FAIL` | One or more mandatory criteria failed | Stop at the failing boundary, add regression coverage, remediate, and rerun |
| `BLOCKED` | An optional dependency or platform is unavailable | Continue only where the stage document explicitly permits it; core gates remain closed |

A passing capability metric cannot override a failed correctness, reproducibility, provenance, or safety criterion. Conversely, a passing infrastructure test cannot substitute for capability evidence.

## Required artifact layout

Each stage run should produce an artifact directory following this convention:

```text
artifacts/stage-{id}/{commit}-{config_hash}/
├── manifest.json
├── config.json
├── environment.json
├── tests.json
├── benchmarks.json
├── gate.json
├── logs/
├── profiles/
├── checkpoints/
└── report.md
```

The artifact manifest must identify which files are generated, which are evaluator-only, and which are safe to publish. Evaluation truth, hidden test data, secrets, and private source material must not be copied into public reports.

## Transition discipline

Stage implementation and stage evaluation are separate activities. Engineers may run smoke tests during development, but the final gate must use a clean checkout, a frozen configuration, declared seeds, immutable test manifests, and a release candidate commit. If the implementation changes after the final gate, the gate is invalidated and must be rerun.

All claims must be supported by a baseline and an ablation. In particular, the project must measure the contribution of the recurrent core, spectral operator, causal metadata, mode workspace, memory, and deliberation independently. A component that fails its ablation should be simplified or removed rather than protected by increasingly complex evaluation exceptions.

## Recommended execution order

The approved order is strictly sequential:

```text
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7
```

Parallel engineering may be used for tooling and documentation, but no capability stage should be declared transitioned until its predecessor has a `PASS` gate. Stage 7 ends in a controlled research review rather than automatic deployment.

## References

[1]: ../CCT_EVOLUTION_PROPOSAL.md "CCT-ASE evolution proposal"

[2]: ../README.md "Current CCT README"

[3]: ../Architecture.md "Current CCT architecture specification"
