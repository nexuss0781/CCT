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
| 6 | [Deliberation and Verification](06_Deliberation_Verification.md) and [Expanded Gate Contract](06_Expanded_Gate_Contract.md) | Native bounded planning, independent verifiers, evidence abstention, deny-by-default offline tools, replay, interruption, and incident logging | Stage 6 gate in progress; Stage 7 requires explicit approval |
| 7 | [Multimodal and Open-Ended Research](07_Multimodal_Open_Ended.md) and [Expanded Gate Contract](07_Expanded_Gate_Contract.md) | Native typed multimodal events, seven adapters, alignment, mask-aware fusion, typed memory, deterministic simulation, transfer, audit, and safety controls | Terminal gate in progress; PASS authorizes controlled continuation only |
| 8 | [Production NLP Roadmap](08_Production_NLP_Roadmap.md) and [Expanded Gate Contract](08_Expanded_Gate_Contract.md) | Native governance registry, policy boundary, realistic application fixtures, artifact protocol, adversarial controls, and readiness evidence | PASS — governance foundation only; Stage 9 requires explicit approval |
| 9 | [Governed Data and Corpus](09_Governed_Data_Corpus.md) and [Expanded Gate Contract](09_Expanded_Gate_Contract.md) | Native real-source manifest, rights/privacy quarantine, exact/near deduplication, contamination barrier, deterministic shards, replay, deletion, and audit | PASS — Stage 10 requires explicit approval |
| 10 | [Tokenizer and Representation](10_Tokenizer_Representation.md) and [Expanded Gate Contract](10_Expanded_Gate_Contract.md) | Native byte/subword/hybrid tokenizer candidates, immutable vocabulary/snapshot, byte fallback, offsets, provenance, packed/padded causal batches, and efficiency comparison | **PASS — Stage 11 requires explicit approval** |
| 11 | [Trainable Native NLP Core](11_Trainable_Native_NLP_Core.md) and [Expanded Gate Contract](11_Expanded_Gate_Contract.md) | Native categorical next-token trainer, analytic CCT recurrence gradients, optimizer/checkpoint recovery, real-source pilot, and matched controls | **PASS — Stage 12 requires explicit approval** |
| 12 | [Scaling and Accelerator Systems](12_Scaling_Accelerator_Systems.md) and [Expanded Gate Contract](12_Expanded_Gate_Contract.md) | Native CPU reference/fused scaling matrix, resource profiling, parity, atomic recovery, and backend decision | **PASS — Stage 13 requires explicit approval** |
| 13 | [Supervised Fine-Tuning and Adapters](13_Supervised_Fine_Tuning_Adapters.md) and [Expanded Gate Contract](13_Expanded_Gate_Contract.md) | Native six-task SFT, full/adaptor comparison, structured outputs, citation/safety retention, permissions, and deletion lineage | **PASS — Stage 14 requires explicit approval** |
| 14 | [Preference Tuning and Alignment](14_Preference_Tuning_Alignment.md) | Native governed preference data, DPO-like alignment, verifier-weighted reranking, adversarial controls, calibration, blind review, and regression evidence | **PASS — Stage 15 requires explicit approval** |
| 15 | [Verified Retrieval and Knowledge](15_Verified_Retrieval_Knowledge.md) | Native typed retrieval, citations, freshness, conflicts, deletion, poisoning isolation, audit, and verified grounding | **PASS — Stage 16 requires explicit approval** |
| 16 | [Production Inference and Operations](16_Production_Inference_Operations.md) | Native serving API, batching, state/cache, observability, SLOs, canaries, and rollback | Requires Stage 15 PASS |
| 17 | [Controlled Pilot and Production Release](17_Controlled_Pilot_Production_Release.md) | Shadow, limited pilot, human oversight, incident response, rollback, and bounded release | Terminal release gate; no automatic Stage 18 |

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
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7 → Stage 8 → Stage 9 → Stage 10 → Stage 11 → Stage 12 → Stage 13 → Stage 14 → Stage 15 → Stage 16 → Stage 17
```

Parallel engineering may be used for tooling and documentation, but no capability stage should be declared transitioned until its predecessor has a `PASS` gate. Stage 7 ends in a controlled research review rather than automatic deployment. The production segment begins only after explicit approval and remains strictly gated through Stage 17; Stage 17 has no automatic successor.

## References

[1]: ../CCT_EVOLUTION_PROPOSAL.md "CCT-ASE evolution proposal"

[2]: ../README.md "Current CCT README"

[3]: ../Architecture.md "Current CCT architecture specification"
