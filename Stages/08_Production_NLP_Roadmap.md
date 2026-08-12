# CCT-ASE Production NLP Roadmap
## Master Stage Index and Universal Gate Protocol

**Project:** Chrono-Causal Tapestry — Adaptive Spectral Engine (CCT-ASE)  
**Roadmap segment:** Production NLP engine  
**Predecessor:** Stage 7 — Multimodal and Controlled Research  
**Scope:** Stages 8–17  
**Implementation constraint:** Native C++20 for project core, trainers, gates, and serving control plane; accelerator bindings may use audited native CUDA/HIP/vendor libraries after explicit review
**Status:** Stage 8 governance foundation implemented and gated; Stages 9–17 remain specification-only

## Purpose

Stages 0–7 established a research prototype. Stages 8–17 define the complete path from that prototype to an evidence-bounded production NLP engine. The roadmap covers governance, data, tokenization, trainable language modeling, scaling, fine-tuning, preference alignment, verified retrieval, serving, and controlled release.

The roadmap does not assume that CCT-ASE will outperform Transformers. Every production candidate must be compared with a matched Transformer control and, where useful, a hybrid CCT–Transformer system under equal data, compute, latency, and evaluation conditions.

## Scope and non-goals

The master stage covers the production NLP program from governance and corpus preparation through bounded release. It does not itself implement training, serving, or deployment; those are specified independently in Stages 9–17. It does not authorize production use merely by being approved.

## Universal transition protocol

Every stage is implemented completely before its gate is run. A stage passes only when all mandatory checks are green, artifacts are reproducible, and no declared limitation is deferred. The repository must record the stage specification, implementation commit, configuration hash, data and environment manifests, hardware/software versions, random seeds, metrics, failures, and final decision.

A `PASS` authorizes preparation of the next stage only. It does not authorize deployment, external actions, unrestricted code execution, online learning, or autonomous operation. A `FAIL` requires remediation and rerun. A `BLOCKED` result is valid when data rights, privacy, security, hardware, or evaluation controls are unresolved; blocked work cannot be represented as capability evidence.

## Stage sequence

| Stage | Name | Primary output | Mandatory transition decision |
|---:|---|---|---|
| 8 | Production foundation and governance | Product boundary, threat model, data policy, experiment/evaluation registry | Approve measurable production program |
| 9 | Governed data and corpus pipeline | Licensed, deduplicated, privacy-filtered, manifest-addressed corpus | Approve training data and held-out splits |
| 10 | Tokenizer and representation engine | Versioned tokenizer, packed batches, token/data contracts | Approve model input contract |
| 11 | Trainable native NLP core | Real next-token trainer, optimizer, checkpoints, matched controls | Approve base-model training |
| 12 | Scaling and accelerator systems | Scaling laws, accelerator path, distributed/recovery systems | Approve selected production architecture |
| 13 | Supervised fine-tuning and adapters | SFT, full fine-tuning, PEFT/LoRA-style adapters | Approve task adaptation |
| 14 | Preference tuning and alignment | Preference data, DPO/RLHF alternatives, safety alignment | Approve behavior release candidate |
| 15 | Verified retrieval and knowledge plane | Retrieval, citations, memory, deletion, grounding | Approve grounded generation |
| 16 | Production inference and operations | Native serving API, observability, quotas, rollback | Approve shadow production |
| 17 | Controlled pilot and production release | Shadow, limited pilot, SLOs, incident and rollback evidence | Approve bounded production deployment |

## Dependency graph

```text
Stage 8 governance
        |
        v
Stage 9 data -----> Stage 10 tokenizer -----> Stage 11 trainer
                                                |
                                                v
                                        Stage 12 scaling/systems
                                                |
                         +----------------------+----------------------+
                         v                      v                      v
                 Stage 13 SFT/adapters  Stage 15 retrieval      Stage 16 serving
                         |                      |                      |
                         v                      +----------+-----------+
                 Stage 14 alignment                        v
                                                Stage 17 controlled release
```

Stages 13 and 15 may develop in parallel after Stage 11, but both must be integrated and evaluated before Stage 17. Stage 16 can prototype earlier, but no production serving gate may pass until a release candidate from Stages 13–15 exists.

## Universal artifact contract

Each stage must write a machine-readable `gate.json`, `checks.json`, `metrics.json`, `manifest.json`, `incident_log.json`, and human-readable `report.md` under `artifacts/stage-N/cpp-gate/`. Training stages additionally write checkpoint metadata, optimizer state metadata, data cursor, configuration, seed record, and resource profile. Data stages additionally write source licenses, privacy classification, transformations, hashes, and split assignments.

## Evaluation harness

The Stage 8 harness validates that every Stage 9–17 specification has an implementation contract, measurable metrics, independent baselines, artifacts, a failure policy, and an approval boundary. It also validates dependency ordering, artifact naming, claim boundaries, and that no stage is marked implemented merely because its Markdown specification exists.

## Mandatory gate checks

| Check | Pass condition |
|---|---|
| Stage completeness | All Stage 8–17 documents exist and define scope, implementation, evaluation, gate, deliverables, and limitations |
| Dependency order | Every successor names a predecessor and no stage bypasses approval |
| Baseline policy | Each capability stage declares a matched control and ablation |
| Artifact policy | Machine-readable and human-readable release artifacts are defined |
| Data governance | Training and evaluation data have separate provenance and access rules |
| Safety boundary | External action, online learning, and unrestricted code execution remain denied by default |
| Reproducibility | Commit, config, data, seed, hardware, and software identity are required |
| Claim boundary | Proposal text separates specification from demonstrated capability |

## Pass/fail transition

Stage 8 passes only when the production program is approved as a measurable sequence and the next stage’s resources, owners, data controls, and release authority are named. A Stage 8 `PASS` authorizes Stage 9 implementation only; it does not authorize model training or deployment.

## Deliverables

The stage must deliver this master index, the ten independent successor specifications, the dependency graph, universal artifact contract, baseline policy, approval protocol, production decision record, and a roadmap completeness report.

## Explicit limitations

This master stage implements only the production planning, governance, registry, policy, application-fixture, and readiness foundation. It does not implement a trainer, tokenizer, governed corpus, production model, serving system, or deployment. Its `PASS` cannot be used as evidence that Stages 9–17 are complete.

## Universal baseline policy

The minimum comparator set is:

| Baseline | Required use |
|---|---|
| Dense causal Transformer | Quality and mature serving reference |
| GRU or recurrent baseline | Recurrent capacity and optimization control |
| Diagonal or selective SSM | Efficient sequence control |
| CCT-ASE | Candidate architecture |
| CCT–Transformer hybrid | Optional but strongly recommended for production selection |

A claim of advantage is valid only for a declared workload and matched budget. A CCT result that does not beat the control must be reported as a limitation, not hidden by selecting a more favorable metric.

## Universal non-claims

This roadmap does not claim general intelligence, consciousness, autonomous research, safe open-ended agency, universal Transformer replacement, or readiness for unrestricted deployment. It defines a disciplined program for producing evidence about a production NLP engine. Any external tool, persistent identity, online learning, real-world action, or autonomous self-modification requires a new threat model and approval gate.

## Terminal outcome

Stage 17 `PASS` authorizes a bounded production release only for the specific use cases, user groups, data classes, model versions, and action permissions named in its release record. It does not authorize expansion to new domains or higher-consequence decisions without a new evaluation and approval cycle.

## References

[1]: ../Production_NLP_Deployment_Roadmap.md "CCT-ASE Production NLP Deployment Roadmap"

[2]: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence "NIST Generative AI Risk Management Profile"

[3]: https://crfm.stanford.edu/helm/ "Stanford Holistic Evaluation of Language Models"
