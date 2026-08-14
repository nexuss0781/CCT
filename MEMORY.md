# CCT Memory and Conversational Context

## Document status

This document records the **implemented memory behavior and declared memory strategy** of the Chrono-Causal Tapestry (CCT) engine. It intentionally separates what the current native C++20 system does from what must be implemented and evaluated before CCT can claim reliable long-term conversation or teaching eligibility.

The current baseline is the English Acquisition release pushed at commit `505e4305cab3867ff9297424272b4d8a91eb3a51`. This document does not change the model or begin teaching implementation.

## Executive summary

CCT currently has a **bounded rolling token context**, a **session-state service layer**, and separate **event-log and knowledge-retrieval contracts**. It does not yet have proven human-like conversational memory, reliable long-term factual recall, automatic semantic memory formation, or online learning from conversations.

The current checkpoint model can condition its next response on recent prior tokens. That is comparable in principle to a causal language model receiving previous conversation tokens, but its demonstrated context contract is only **256 tokens**. When the rolling window is full, older tokens are removed. The model does not retain a persistent hidden state across inference calls; the runtime retains token IDs and replays the bounded context through the model.

The current service also keeps tenant-, user-, and session-scoped runtime bookkeeping. This includes a bounded token context and a transcript digest. A digest is an audit identity, not recoverable conversation content. The default session time-to-live is five minutes, and this state is in-memory rather than a durable cross-process user memory store.

The repository contains a separate memory and knowledge plane with versioned records, event-log replay, retrieval, citations, provenance, tombstones, quarantine, and snapshots. Those facilities are **not equivalent to the model remembering a conversation**. They must be explicitly connected to dialogue state and checkpoint inference, then evaluated for retrieval correctness, privacy, update, deletion, conflict, and grounding behavior.

## Four distinct memory layers

| Layer | Current CCT behavior | Retention boundary | Current interpretation |
|---|---|---|---|
| Model-weight memory | Training updates checkpoint parameters using next-token and preference objectives. | Persists in the checkpoint until a new checkpoint replaces or adapts it. | Statistical learned behavior, not a verified factual database and not online user memory. |
| Active token context | Checkpoint inference combines prior session token IDs with the current input and performs bounded greedy decoding. | Rolling model context is capped at 256 tokens in the released Track 1 configuration. | Short-term causal context; older tokens are dropped when the window fills. |
| Runtime session state | `InferenceService` stores tenant/user/session identity, transcript digest, and optional model context. | Default state TTL is 300,000 milliseconds; state is resettable and evictable. | In-memory service bookkeeping and recent-token continuation, not durable semantic memory. |
| External memory and knowledge | Separate typed records, event log, retrieval, provenance, citations, conflict/tombstone/quarantine states, and snapshots are available. | Governed by record validity, retention, deletion, and snapshot contracts. | A potential durable memory/grounding substrate; not automatically used by the language model for every conversation. |

## How CCT answers after training

The released checkpoint inference path operates as follows:

1. The tokenizer encodes the new input.
2. The runtime appends that input to the prior token context for the session.
3. If the combined context is longer than the model limit, the oldest tokens are removed.
4. The Track 1 CCT recurrence processes the bounded token sequence causally.
5. The highest-logit next token is selected greedily.
6. The new token is appended to the rolling context and generation continues until EOS, the output limit, cancellation, or a resource boundary.
7. The resulting token context is stored back into the session runtime state.

The recurrent hidden state is not currently serialized as an independently persistent conversational memory between requests. Each `next_logits` evaluation reconstructs the recurrent state by processing the supplied token context. This is deterministic and simple to audit, but it means that effective memory is bounded by the retained token window and the model’s ability to compress information into that window.

## What is retained and what is not

### Retained within an active session

Recent user and assistant tokens can influence later responses while they remain inside the rolling context limit. The session identity prevents one tenant/user/session state from being silently reused for another. The service can reset or evict state, and it records audit-oriented identity information.

### Encoded in model weights

Training can encode recurring language patterns, word relationships, syntax, and statistical associations in the checkpoint parameters. It may also encode fragments of factual regularities present in the training corpus. However, the current model is small, the training corpus is bounded, and factual recall has not been evaluated as a reliable knowledge capability. A parameter association must not be treated as a sourced fact.

### Available through external memory and knowledge

A fact may be stored as a governed record with content, embedding or indexed terms, provenance, validity, confidence, retention, and conflict metadata. A later request may retrieve that record and bind a citation. This is the correct direction for factual continuity, but the current release does not prove a complete end-to-end pipeline in which all relevant past conversation facts are automatically extracted, stored, retrieved, injected into the model, verified, and deleted according to user policy.

### Not currently retained reliably

The current release does not yet guarantee durable cross-session user preferences, semantic summaries of arbitrarily long conversations, automatic fact extraction from every turn, contradiction-resolved personal memory, user-controlled memory editing, cross-process memory recovery, or safe online weight updates from conversation. It also does not guarantee that a fact present in old context will be recalled once it has fallen outside the 256-token window.

## Memory versus factual knowledge

Memory and factual knowledge must remain separate:

| Question | Correct CCT answer today |
|---|---|
| Can recent text affect the next answer? | Yes, within the bounded rolling token context. |
| Does the model remember every previous turn? | No. Older tokens are dropped at the context boundary. |
| Does a session digest preserve the conversation? | No. It supports identity/audit checks and cannot reconstruct the text. |
| Does training create a reliable fact database? | No. Weights encode statistical behavior; factual reliability is unproven. |
| Can the repository store governed facts separately? | Yes, through its memory/knowledge contracts, subject to explicit integration and evaluation. |
| Does CCT learn permanently from each live conversation? | No. Online learning is outside the current release contract. |
| Can CCT currently claim ChatGPT-level long-term conversation? | No. Multi-turn state, grounding, calibration, and human-quality evidence are still required. |

## Memory strategy before teaching

The memory strategy is staged and must be implemented only with explicit schemas, deterministic replay, and failure-path gates.

| Memory stage | Required capability | Required evidence before transition |
|---|---|---|
| M0: Memory contract | Define memory classes, ownership, scope, consent, retention, expiry, deletion, provenance, and conflict policy. | Versioned schema, identity hash, fail-closed policy, reset/delete tests. |
| M1: Short-term context | Use role-aware turn formatting, bounded context management, context compression or summarization, and deterministic overflow behavior. | Long-context replay, truncation audit, contradiction tests, latency and memory measurements. |
| M2: Dialogue state | Track user goal, entities, references, constraints, unresolved questions, corrections, confidence, and state expiry separately from raw tokens. | Per-turn state accuracy, correction/recovery tests, state replay, topic-shift and ambiguity tests. |
| M3: Durable personal memory | Store only explicitly permitted facts or preferences with user/session scope, provenance, confidence, expiry, deletion, and isolation. | Cross-session recall, consent, deletion, conflict resolution, leakage, corruption, and recovery tests. |
| M4: Grounded factual memory | Retrieve relevant records and evidence before answering; distinguish stored fact, current retrieval, inference, and unknown. | Claim-level support, citation correctness, stale/conflicting evidence, abstention, and adversarial retrieval tests. |
| M5: Controlled learning | Update checkpoints only through governed offline training or explicitly approved adaptation jobs. | Base immutability, data approval, contamination controls, rollback, reproducibility, and regression gates. |
| M6: Conversational eligibility | Combine memory with ambiguity handling, instruction following, coherence, calibration, and multi-turn task success. | Independent interactive episodes, held-out domains, blind human evaluation, and no unresolved critical failure. |

## Design principles

### Explicit memory is safer than implicit guessing

If a fact is important, the system should represent it as a typed memory record or dialogue-state field with provenance and confidence. The model must not silently infer that a fluent continuation is a confirmed user fact.

### Context is not memory

A token window is temporary working context. Durable memory requires an explicit record, retention rule, identity, and retrieval path. Summaries are also memory artifacts and must carry provenance and uncertainty rather than silently replacing the original conversation.

### Retrieval is not truth

A retrieved record is evidence to inspect, not an unconditional answer. The system must handle stale, conflicting, incomplete, unauthorized, or low-confidence records and abstain when the evidence does not support a claim.

### Training is not live memory

Conversation logs must not silently become training data. Any learning update requires separate data governance, consent and policy review, contamination controls, checkpoint lineage, reproducible training, rollback, and evaluation against prior behavior.

### Forgetting must be testable

A correct memory system must support reset, expiry, deletion, and scope isolation. The gate must verify not only that allowed facts are recalled, but also that deleted or unauthorized facts are not recalled later.

## Current eligibility decision

CCT is currently **not memory-eligible for open-ended conversational teaching**. It has enough short-term context machinery to begin a dedicated conversational-readiness qualification, but it must first demonstrate structured dialogue state, ambiguity-aware clarification, grounded factual retrieval, calibration, and durable-memory controls within a declared scope.

The immediate technical gate remains the architecture-and-trainability qualification. Memory expansion should not be used to hide a weak language model, and a larger context window should not be treated as proof of understanding. After architecture qualification, the project should proceed through short-term context, dialogue state, ambiguity, grounding, and then durable memory gates before teaching behavior.

## Implementation references

The current behavior is grounded in these repository interfaces and implementations:

| Component | Role |
|---|---|
| `cpp/include/cct/nlp_trainer.hpp` | Model context contract, checkpoint model, optimizer, and evaluation interfaces. |
| `cpp/src/nlp_trainer.cpp` | Track 1 recurrence, `next_logits`, bounded context validation, checkpoint serialization, and deterministic training. |
| `cpp/include/cct/inference.hpp` | Inference request, session, backend, runtime-state, memory-budget, and policy contracts. |
| `cpp/src/inference.cpp` | Checkpoint-backed greedy decoding, rolling token context, session updates, retrieval routing, streaming, and state eviction. |
| `cpp/include/cct/memory.hpp` and `cpp/src/memory.cpp` | Event-log memory records, replay, tombstones, quarantine, retrieval, citations, and snapshots. |
| `cpp/include/cct/knowledge.hpp` and `cpp/src/knowledge.cpp` | Versioned knowledge records, retrieval, grounding, and citation verification. |
| `Architecture.md` | Implementation-grounded architecture boundaries and non-claims. |
| `SPEC/Goal.md` | Level 1 language, instruction, operations, teaching, and reliability sequencing. |

## References

[1]: https://crfm.stanford.edu/helm/ "Stanford HELM"

[2]: https://arxiv.org/html/2503.22458v1 "Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey"

[3]: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737/128713 "Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph"

[4]: https://arxiv.org/html/2409.06097v2 "ClarQ-LLM"
