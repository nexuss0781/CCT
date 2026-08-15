# CCT Focused-English Competency Curriculum and Continual Learning

## Purpose

This specification changes the next CCT training path from one uninterrupted training run into a sequence of auditable competency sessions. Each session selects a deterministic document range from a pinned focused-English source, trains from the previous session checkpoint, evaluates held-out material, produces a human mastery packet, and stops. The next session is not allowed to begin until the user records a signed result for the current packet.

The system is intended to answer a narrower and more useful question than a single aggregate loss: **which declared English competency has this architecture demonstrated on unseen material, and where does it fail?** A failed competency is evidence for investigation, not proof by itself of an architectural defect. The diagnosis becomes architectural only after data identity, tokenizer identity, optimizer state, checkpoint lineage, validation procedure, and reproducibility checks all pass.

## Dataset selection

The initial focused-English pretraining source is the English educational subset `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, split `train`, acquired through the official datasets-server rows endpoint using fixed `offset` and `length` ranges. The source exposes stable document identifiers, text, URL, language, language score, token count, and educational score fields. The first implementation will use small deterministic ranges from this large source rather than downloading the entire corpus.

The supervised source is `OpenAssistant/oasst1`, configuration `default`, split `train`, acquired through the same rows endpoint. The native preparer retains English rows, non-deleted messages, valid conversation-tree identifiers, reviewable examples, and assistant responses with stable parent-child links. SQuAD-style answer-target examples remain a secondary comprehension slice for explicit answer extraction and abstention checks.

GLUE-style sentence and sentence-pair tasks are evaluation-only competency material. They must not be mixed into pretraining or SFT. Dolma is reserved for a later scale-up because its Ai2 ImpACT license imposes access, derivative, and use obligations that require a dedicated native manifest and license workflow before ingestion.

Every source record must retain its upstream dataset ID, configuration, split, revision or source digest, row offset, row index, stable document/message ID, language filter result, quality filter result, and chunk membership. Training, selection validation, human-test prompts, and later sessions must be disjoint by stable source ID.

## Competency ladder

| Level | Competency | Training emphasis | Mastery evidence supplied to the user |
|---|---|---|---|
| 0 | Stable English symbols and word boundaries | Short educational prose, punctuation, whitespace, common words | Exact continuation, token validity, punctuation and word-boundary prompts |
| 1 | Sentence completion and grammatical local coherence | Complete sentences and adjacent sentence contexts | Unseen sentence completions, agreement, tense, articles, and punctuation |
| 2 | Paragraph coherence and topic persistence | Multi-sentence educational documents with controlled context | Unseen paragraph continuation and topic-maintenance prompts |
| 3 | Reading comprehension and answer targeting | English assistant examples plus answer-target records | Extractive answers, unanswerable questions, and abstention prompts |
| 4 | Instruction following and response structure | Filtered English OASST conversation trees | User-provided instruction prompts with format, relevance, and completeness checks |
| 5 | Ambiguity recognition and clarification | Multi-turn English prompts with ambiguous or underspecified requests | Ambiguous prompt set requiring clarification before answering |
| 6 | Conversational continuity and repair | Disjoint multi-turn conversation chunks and correction examples | Follow-up, correction, contradiction, and context-retention prompts |
| 7 | Transfer and bounded generalization | Held-out mixtures of prior competencies | New-domain English prompts; failure blocks progression and triggers diagnosis |

The first implementation must complete Levels 0–2 before attempting conversational work. Levels 3–7 are specified but remain blocked until the lower-level generation and comprehension behavior is demonstrated on unseen user-reviewed material.

## Session contract

A session has a stable ID, curriculum level, source chunk ranges, tokenizer hash, model configuration, optimizer configuration, parent checkpoint hash, training seed, training step budget, validation split identity, and output directory. Its checkpoint is immutable after publication. The session report must contain before/after held-out metrics, checkpoint hash, parent lineage, selected row IDs and counts, document/token counts, elapsed time, and the human-validation status.

The run script has four fail-closed states:

| State | Meaning | Allowed action |
|---|---|---|
| `READY_TO_TRAIN` | No unresolved session is present and the curriculum state identifies the next level | Train exactly one session |
| `AWAITING_HUMAN_VALIDATION` | The session checkpoint and mastery packet are complete | Stop; wait for the user’s result |
| `MASTERED` | A human result marked the current session mastered | Advance the curriculum pointer and permit the next session on the next invocation |
| `ARCHITECTURE_DIAGNOSIS_REQUIRED` | Human failure or invariant failure occurred | Stop; preserve all artifacts; do not silently change data or architecture |

The script must never automatically mark a competency mastered from loss, perplexity, token accuracy, or a generation heuristic. Automatic metrics are supporting evidence only. The user must create a validation record containing the session ID, checkpoint hash, competency level, result (`PASS` or `FAIL`), evaluator name or identifier, timestamp, prompt-by-prompt observations, and an optional diagnosis note. A `PASS` advances one level. A `FAIL` stops the curriculum and writes an architecture/data diagnosis packet.

## Checkpoint and lineage requirements

The loaded checkpoint must match the frozen tokenizer hash and model identity. Continuation onto a new chunk must retain model parameters, optimizer moments, global optimizer step, deterministic data cursor, seed, and parent checkpoint hash while binding the new session to a new dataset identity. The new dataset identity must not be substituted into an old checkpoint silently; lineage must explicitly record both the parent dataset hash and the current session dataset hash.

A session is publishable only when checkpoint serialization, atomic replacement, hash computation, reload, parent-hash verification, model-parameter equality after reload, and deterministic one-step replay all pass. Missing, malformed, stale, or mismatched human validation records must block advancement.

## Transition rule

A failed session does not authorize trying a larger dataset or changing the architecture in place. First preserve the failed checkpoint and session packet. Then repeat the same competency with a fresh disjoint chunk under the same contract. If the repeat fails while data and protocol controls pass, the report may classify the result as an architecture qualification failure and stop. If the repeat passes, the prior failure is classified as chunk variance and the curriculum may continue with the evidence retained.

This curriculum is a research qualification protocol. It does not claim broad language competence, human-speaker equivalence, factual reliability, safe deployment, or general intelligence.
