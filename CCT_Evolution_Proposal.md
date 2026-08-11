# CCT-X: Proposal for an Efficient, Trainable Intelligence Engine Beyond Transformers

**Prepared for approval by Manus AI**  
**Repository assessed:** [nexuss0781/CCT](https://github.com/nexuss0781/CCT)  
**Status:** Proposal only; no project modifications are proposed before approval.

## Executive decision

I propose evolving the Chrono-Causal Tapestry (CCT) into **CCT-X**, a hybrid **causal state-space and spectral neural-operator engine**. The main path will not use quadratic self-attention. It will combine four mechanisms: a selective recurrent state, a learned spectral field operator, a sparse causal event substrate, and a durable read/write memory.

The important correction is that CCT should not attempt to turn every speculative mathematical idea into a production primitive at once. The repository currently contains a promising vocabulary and specification, but it is still a prototype: the Rust core has only minimal event placement and lookup, `create_source_field` returns an all-zero field, `objective.py` and `resonance.py` are empty, and the benchmark section is still marked “Coming soon.” The existing tests cannot currently be run in this environment because `pytest` is unavailable, and the Rust core cannot be compiled here because `cargo` is unavailable. These are baseline facts, not failures of the underlying idea.

The project should therefore proceed through falsifiable gates. The goal is not to claim “superintelligence” by architecture name. The goal is to build a reproducible engine that can demonstrate **general-purpose sequence competence, persistent memory, causal state tracking, compositional reasoning, and substantially better efficiency at matched quality**. Only after those properties are independently measured should the project make stronger claims about intelligence.

> **Proposed approval boundary:** approve implementation of the staged CCT-X research program, beginning with a buildable numerical and training baseline. Do not approve uncontrolled autonomous operation, self-modification, large-scale training expenditure, or external deployment until the stated evaluation and safety gates pass.

## 1. What CCT is today

CCT’s documentation proposes causal manifolds, spectral propagation, geometric semantics, resonant modes, and topological memory as an alternative to Transformer attention. Its Phase 1 specification calls for a sparse octree/quadtree substrate with causal queries, while Phase 2 calls for differentiable spectral solving, and Phase 4 calls for a mode-coupling reasoning engine. Those are useful architectural contracts, but most of those layers do not yet exist in executable form.

| Area | Current repository state | Consequence for CCT-X |
|---|---|---|
| Event representation | Minimal semantic vector, coordinates, and causal potential; no timestamp or causal-parent graph in the active public implementation | Add validated event IDs, timestamps, parent links, masks, and batch serialization |
| Manifold substrate | Dense `ndarray::ArrayD<Option<Event>>`; only placement, exact lookup, and representation | Replace with sparse, batched storage and explicit causal/range/k-nearest queries |
| Spectral computation | A small FFT filter and linear propagation function | Convert to a real differentiable operator with stable dynamics, trainable parameters, and time evolution |
| Source construction | `create_source_field` is explicitly a placeholder and returns zeros | Implement a zero-copy or batched event-to-field encoder |
| Training | No complete model, loss, optimizer loop, checkpointing, or dataset pipeline | Build a pure JAX/Flax training surface with deterministic experiments |
| Resonance engine | `resonance.py` is empty; mode-coupling exists only in specification text | Implement a low-rank mode bank and controlled bilinear coupling |
| Topological memory | Specification only; no executable memory store | Start with a practical causal key/value memory and add topology as an auxiliary signature |
| Validation | A few unit tests for storage and FFT superposition | Add numerical analysis, gradient checks, synthetic reasoning, long-context, memory, and systems benchmarks |
| Complexity claims | Documentation states broad O(n log n) guarantees that do not hold for every irregular manifold or topological operation | Replace universal claims with operation-specific measured complexity bounds |

The first engineering principle is therefore **make the system real before making it large**. A smaller, fully tested CCT-X will provide more scientific value than a large system whose geometric and topological claims cannot be measured.

## 2. Proposed CCT-X architecture

CCT-X should be a **five-part engine** with a clean separation between representation, dynamics, memory, and task output. The main sequence path remains linear in sequence length for recurrent processing; global spectral work is invoked through bounded-size mode spaces or regular-grid FFTs rather than token-by-token pairwise attention.

```text
Input events/tokens
        |
        v
[Validated Event Encoder]
        |
        +--> [Sparse Causal Event Graph] ----> [Causal Read/Write Memory]
        |
        v
[Selective Complex State-Space Recurrence]
        |
        +--> [CCT Spectral Field Operator]
        |          |
        |          +--> low-rank learned modes
        |          +--> stable wave/diffusion dynamics
        |          +--> optional irregular-graph operator
        |
        v
[Mode Coupling + Gated Workspace]
        |
        v
[Task heads / decoder / tool policy]
```

### 2.1 Validated event and token substrate

The active `Event` structure should be replaced with a typed record containing an `EventId`, logical timestamp, source modality, semantic payload, spatial or manifold coordinate, causal-parent IDs, and validity masks. Insertion should reject malformed coordinates, duplicate IDs, invalid parent references, and violated causal ordering. The Rust layer should own storage and indexing; JAX should receive dense batches of event attributes rather than attempting to iterate over Rust objects inside a JIT-compiled function.

The sparse manifold should not be treated as a universal dense tensor. It should use an arena-backed event store plus a spatial index, with separate indexes for timestamp, parent/child links, and optional geometric coordinates. Causal queries should return compact event IDs and gatherable feature arrays. This makes the substrate useful for streaming data, while retaining the repository’s causal-manifold intent.

### 2.2 Selective complex state-space recurrence

The backbone should be a **selective state-space recurrence** rather than self-attention. At each event, the input controls the state transition, write strength, readout, and forgetting rate. This is the practical mechanism that allows a linear-time model to decide which information to retain instead of applying a fixed convolution to every token. The design is consistent with the direction of selective state-space models, which were proposed to address content-based reasoning limitations in earlier subquadratic models while retaining linear scaling [1].

CCT-X should extend this recurrence with three CCT-specific features:

1. **Causal coordinates:** the transition step is a function of event time, causal depth, and optional manifold distance rather than only token position.
2. **Complex or paired real state:** oscillatory modes can preserve phase and support state tracking without requiring a dense attention matrix.
3. **Multi-input/multi-output updates:** several feature channels update the state together to improve hardware utilization and expressivity.

The 2026 Mamba-3 results are a useful engineering reference rather than a component to copy blindly: its peer-reviewed description combines more expressive discretization, complex-valued state, and MIMO updates to improve retrieval, state tracking, language modeling, and the performance–efficiency tradeoff [4]. CCT-X should use those lessons while making the transition operator geometry-aware and testable.

### 2.3 CCT spectral field operator

The current FFT function is a useful numerical seed, but the present implementation is only a linear filter. It needs to become a **stable, trainable operator block**:

\[
 h_{t+1} = \mathcal{D}_{\theta}(h_t, x_t, m_t), \qquad
 y_t = P_{\theta}(h_t),
\]

where \(\mathcal{D}_{\theta}\) is implemented either as a spectral recurrence on a regular grid or as a sparse graph/neural operator on irregular events. The operator should use a bounded potential, positive damping, explicit step-size parameterization, and a stability monitor. The model should never rely on an unbounded learned denominator or unconstrained physical parameter.

For regular grids, FFT propagation may achieve O(N log N) in the number of grid cells. For irregular manifolds, the implementation should use sparse graph operators, Chebyshev polynomial filters, or a learned low-rank mode basis; it must not claim that arbitrary irregular geometry is automatically O(N log N). The topological memory path must likewise be described using its actual approximate or worst-case complexity, because exact persistent homology can be much more expensive than the current README suggests.

### 2.4 Learnable low-rank resonance bank

The resonance engine should begin with a **learned mode bank**, not a full differentiable eigendecomposition at every training step. Let \(\Psi \in \mathbb{R}^{N \times m}\) contain \(m \ll N\) modes. The engine projects the field into mode space, evolves mode amplitudes, applies gated linear and bilinear coupling, and projects back:

\[
 a_t = \Psi^\top h_t, \qquad
 a_{t+1} = A(a_t, x_t) + B(a_t \otimes a_t), \qquad
 \tilde h_t = \Psi a_t.
\]

The mode basis can be initialized from a fixed Laplacian or Fourier basis and later fine-tuned with orthogonality and smoothness regularizers. This makes the resonance claim measurable: we can determine whether a small number of modes actually preserves task performance, whether mode coupling improves compositional tasks, and whether the learned modes are stable across seeds.

The mode bank is inspired by the repository’s Phase 4 idea of replacing token-wise attention with mode-space interaction. It also incorporates an important lesson from Hyena: long convolutions plus data-controlled gating can provide a credible dense-attention-free alternative for long-range recall and reasoning, but the result must be established by matched experiments rather than assumed from the operator’s form [2].

### 2.5 Practical causal memory with topology as an auxiliary signal

Persistent homology should not be the sole memory index. It is too compressed to identify arbitrary content by itself. CCT-X should first implement a practical memory with:

| Memory component | Function | Trainable interface |
|---|---|---|
| Event log | Immutable, timestamped record of observations and actions | Append-only write with provenance |
| Causal graph | Parent/child and temporal relationships | Sparse edge updates and causal traversal |
| Content key/value store | Retrieve semantically relevant episodes | Learned key, value, confidence, and age |
| Consolidated state | Compress repeated or related events | Periodic differentiable read and gated write |
| Topological signature | Describe connectivity and persistence changes | Auxiliary feature and diagnostic, not sole key |

The memory reader should be trained on explicit retrieval and update tasks. The architecture can borrow the decoupled memory principle from LongMem, where a memory encoder and adaptive retriever/reader support long histories without forcing all history through the fixed-size input window [3]. CCT-X’s difference is that memory entries also carry causal provenance and manifold coordinates.

## 3. Changes required in the repository

The work should be organized as a controlled rewrite of the implementation surface while preserving the mathematical documents as design references. The first change is to separate **specification**, **reference implementation**, and **accelerated implementation**. Every claimed property must have a small reference version before a Rust or GPU optimization is accepted.

| Workstream | Required change | Completion condition |
|---|---|---|
| Repository correctness | Fix the Python package loader, add a reproducible environment file, add CI, and make the native module build optional for pure-Python tests | A clean checkout can run CPU tests and report missing optional accelerators clearly |
| Rust substrate | Introduce validated `EventId`, timestamps, causal parents, sparse storage, batch extraction, remove/query/range/causal/k-nearest APIs, and structured errors | Property tests cover all public operations and Python can retrieve batched event tensors |
| Data model | Define a versioned event schema with masks, modality IDs, coordinate systems, and provenance | Serialization round-trips exactly and malformed events are rejected |
| Spectral module | Replace the one-off Gaussian kernel with `CCTOperator`, explicit boundary handling, stable time stepping, learnable potential/damping, and `jax.lax.scan` evolution | Analytic PDE, convergence, energy, and gradient tests pass |
| Trainable model | Add `CCTBlock`, `CCTBackbone`, task heads, optimizer, checkpointing, mixed precision, and deterministic seeds | A toy language model learns and reproduces results within predefined variance |
| Resonance module | Implement mode initialization, orthogonalization, projections, mode dynamics, and bounded bilinear coupling | Mode ablations and mode-count scaling show measurable behavior |
| Memory module | Add event log, causal graph, key/value retrieval, consolidation, and topology feature adapter | Long-history retrieval and update benchmarks are reproducible |
| Benchmark harness | Add matched baselines, parameter/FLOP accounting, wall-clock measurement, peak-memory measurement, and statistical aggregation | Every model comparison reports quality, throughput, latency, memory, and seeds |
| Documentation | Rewrite broad “guarantee” language into conditional, operation-specific claims | README claims are traceable to tests or labeled as hypotheses |

### 3.1 Software structure

A practical target layout is:

```text
causa_py/
  data/              # event schema, batching, serialization
  models/            # CCTBlock, backbone, heads
  operators/         # spectral, graph, recurrent operators
  memory/            # event log, retriever, consolidation, topology features
  training/          # objectives, optimizer, checkpointing, distributed loops
  evaluation/        # benchmark adapters and statistical reports
  tests/             # unit, property, numerical, gradient, regression tests
causa_core/src/
  event/             # validated records and IDs
  index/             # sparse spatial and causal indexes
  memory/            # arena storage and batch extraction
  bindings/          # PyO3 interface
```

The key design rule is that **JAX functions receive arrays, not opaque Rust objects**. The Rust/Python boundary should be crossed in batches, and the hot numerical path should be pure and JIT-compatible.

## 4. Training strategy

CCT-X should be trained in stages rather than attempting end-to-end superintelligence from an unverified field solver. Each stage teaches a capability that corresponds to one architectural promise.

| Training stage | Objective | What it proves |
|---|---|---|
| Numerical pretraining | Reconstruct known dynamical systems and operator trajectories | The spectral and recurrent dynamics are numerically correct |
| State-tracking pretraining | Copy, selective copy, associative recall, parity, delayed recall, and distractor tasks | The recurrent state can retain, overwrite, and retrieve information selectively |
| Causal event pretraining | Predict valid descendants, identify parents, estimate intervention effects in controlled causal graphs | The event substrate is used causally rather than as decorative metadata |
| Language or multimodal sequence training | Next-event/token prediction plus denoising and masked reconstruction | The engine can learn distributed representations from real data |
| Memory-augmented training | Retrieve relevant episodes, update memories, resist stale or conflicting records | The memory system is useful and provenance-aware |
| Reasoning fine-tuning | Compositional tasks, tool-free planning traces, and verifiable intermediate states | Mode coupling and workspace operations contribute beyond memorization |

The primary objective should be a weighted combination of prediction loss, state-tracking loss, causal consistency loss, memory retrieval loss, stability regularization, and sparsity/compute regularization. The model should not be rewarded for producing explanations that are not checked. Wherever possible, reasoning outputs should be evaluated against executable state transitions, held-out facts, or exact answers.

A proposed loss is:

\[
\mathcal{L} = \mathcal{L}_{\text{predict}}
+ \lambda_s\mathcal{L}_{\text{state}}
+ \lambda_c\mathcal{L}_{\text{causal}}
+ \lambda_m\mathcal{L}_{\text{memory}}
+ \lambda_e\mathcal{L}_{\text{energy/stability}}
+ \lambda_r\mathcal{L}_{\text{regularization}}.
\]

The weights must be tuned only on validation data. The stability and causal terms should be designed so they do not force the model to imitate a physically incorrect prior merely because the mathematics is elegant.

## 5. How the engine will be tested

Testing must be layered. Passing a language benchmark cannot compensate for an unstable solver, and passing a PDE test cannot establish reasoning ability. Every reported result should include at least three random seeds, a matched parameter budget, a matched data budget, hardware details, wall-clock time, peak memory, and confidence intervals or standard deviations.

### Gate A: build and numerical correctness

The first gate tests the substrate and operator independently of language. It should include event-schema validation, insertion/removal/query properties, causal transitivity, serialization, sparse index correctness, analytic wave-packet solutions, spatial and temporal convergence, energy drift, boundary-condition behavior, NaN/Inf handling, and finite-difference versus automatic-differentiation gradient checks. The Phase 2 specification already provides useful targets such as analytic error below 10⁻³, convergence matching the chosen integrator order, and gradient checks near 10⁻⁵ [CCT Phase 2]. Those targets should be retained only after correcting any test formula errors and validating them on the chosen discretization.

### Gate B: capability and ablation tests

The model must be compared with at least three baselines: a small Transformer, a standard recurrent/linear state-space baseline, and a CCT-X ablation without the spectral branch. The central ablations are recurrence only, recurrence plus spectral operator, recurrence plus memory, recurrence plus mode coupling, and the full system.

| Capability | Required test | Pass condition |
|---|---|---|
| Selective retention | Selective-copy and distractor tasks at increasing sequence lengths | Accuracy remains stable as distractors grow, with no quadratic hot path |
| State tracking | Parity, counters, associative recall, and delayed state updates | CCT-X must beat its non-complex and non-selective ablations, not merely fit training data |
| Long-range retrieval | Needle/associative recall and book-length chapter-break style tasks | Performance remains competitive as context exceeds the recurrent training window |
| Composition | CLUTRR-style relational reasoning and held-out composition patterns | Generalization to unseen combinations, not only seen templates |
| Causal structure | Interventions on held-out synthetic structural causal models | Correct counterfactual/interventional ranking with explicit provenance |
| Memory | Write, retrieve, update, conflict resolution, and stale-memory tests | High retrieval precision and correct preference for newer or better-provenanced facts |
| Robustness | Corruption, reordered irrelevant events, missing events, and adversarial distractors | Graceful degradation and no silent memory/provenance violations |

### Gate C: efficiency and scaling

The efficiency claim must be measured against matched-quality baselines. The minimum reporting set is training tokens/events per second, inference tokens/events per second, prefill latency, decode latency, peak memory, checkpoint size, energy or GPU-hours where available, and quality per unit of compute.

The initial targets should be treated as **go/no-go hypotheses**, not guaranteed results:

| Metric | Initial target for CCT-X |
|---|---|
| Main-path complexity | No O(T²) operation in the hot path; recurrent decode memory independent of total history except for the explicit memory store |
| Quality at equal parameter count | Within 5% relative of the strongest matched baseline on general sequence tasks before adding memory |
| Long-context quality | No more than 5 percentage points below the strongest baseline on retrieval/state-tracking suites, with a target to exceed it after memory training |
| Efficiency | At least 30% lower peak memory or 1.5× higher throughput at matched quality on the selected hardware |
| Scaling | Measured empirical slope consistent with linear recurrent processing and documented deviations for spectral or memory operations |
| Reproducibility | Three independent seeds with predefined variance bounds and published configurations |

If CCT-X cannot meet these targets, the design should be narrowed rather than protected by changing the metric after the fact.

### Gate D: systems and safety validation

Before any external tool use or persistent deployment, the model must run in a sandbox with explicit resource limits, immutable logs, human approval for side effects, and a disabled self-modification path. The evaluation should include prompt-injection resistance for retrieved memories, provenance preservation, unauthorized write attempts, data exfiltration tests, reward-hacking checks, and shutdown/interruptibility tests. Memory retrieval must not silently become an authority channel: the system should expose source, timestamp, confidence, and conflict status for every retrieved item.

## 6. What the final evolution should mean

The final goal of this evolution should be stated precisely:

> **CCT-X is a reproducible, beyond-Transformer intelligence engine that learns and maintains structured causal state over very long streams, combines local recurrent dynamics with global spectral interaction, retrieves persistent memories with provenance, performs verifiable compositional reasoning, and delivers a better quality–latency–memory frontier than matched Transformer and state-space baselines.**

This is a credible research objective. It is stronger than “a faster Transformer” because the system is designed around persistent causal state, explicit memory, and operator dynamics. It is more honest than declaring superintelligence in advance. If the engine eventually demonstrates broad transfer, reliable planning, robust self-correction, and efficient inference across modalities and environments, it could become a foundation for increasingly general systems. Those capabilities must be discovered through evaluation, not assumed from the architecture.

## 7. Recommended implementation order after approval

The implementation should begin with the smallest end-to-end vertical slice: validated events, batched extraction, a pure-JAX selective recurrence, one stable spectral operator, one memory store, one task head, and a complete test/benchmark command. Only after that slice passes should the project add geometric metric learning, adaptive manifold refinement, topological summaries, distributed training, or larger models.

| Order | Deliverable | Approval gate |
|---|---|---|
| 1 | Repair packaging, environment, CI, and baseline tests | Clean CPU checkout is reproducible |
| 2 | Implement validated sparse event substrate and batch API | Phase 1 contract is executable |
| 3 | Implement stable spectral operator and differentiable recurrence | Phase 2 numerical and gradient gates pass |
| 4 | Implement CCT-X block and train on synthetic state tasks | Capability ablations show a real contribution |
| 5 | Implement causal memory and provenance-aware retrieval | Memory tests pass without hidden context leakage |
| 6 | Run matched long-context and language benchmarks | Quality/efficiency targets are measured |
| 7 | Add optional geometry/topology modules only if ablations justify them | Complexity and benefit are demonstrated |
| 8 | Scale, optimize, and consider deployment | Safety and reproducibility gates pass |

**Requested decision:** approve or reject the CCT-X architecture and staged validation plan. If approved, the first coding milestone should be limited to steps 1–3; later capability and scaling work should require the numerical gate to pass.

## References

[1]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

[2]: https://arxiv.org/abs/2302.10866 "Hyena Hierarchy: Towards Larger Convolutional Language Models"

[3]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/ebd82705f44793b6f9ade5a669d0f0bf-Abstract-Conference.html "Augmenting Language Models with Long-Term Memory"

[4]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html "Mamba-3: Improved Sequence Modeling using State Space Principles"

[CCT Phase 2]: https://github.com/nexuss0781/CCT/blob/main/SPEC/Phase-2.md "CCT Phase 2: Spectral Solver Specification"

[CCT Phase 1]: https://github.com/nexuss0781/CCT/blob/main/SPEC/Phase-1.md "CCT Phase 1: Manifold Substrate Specification"

[CCT Phase 4]: https://github.com/nexuss0781/CCT/blob/main/SPEC/Phase-4.md "CCT Phase 4: Resonance Engine Specification"

[CCT README]: https://github.com/nexuss0781/CCT/blob/main/README.md "CCT README"

[CCT Architecture]: https://github.com/nexuss0781/CCT/blob/main/Architecture.md "CCT Architecture"
