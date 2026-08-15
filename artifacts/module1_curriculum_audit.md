# Module 1 Curriculum Implementation Audit

## Audit scope

The audit compares the current continual-learning implementation with the approved academic Module 1 path:

1. English symbols, words, and boundaries.
2. Character and symbol awareness.
3. Whitespace and word boundaries.
4. Common word patterns.
5. Stable short continuation.

## Current strengths

- `run.sh` already persists curriculum state, data, session, validation, and reports under `CURRICULUM_ROOT`, which is compatible with Google Drive persistence.
- The existing workflow selects disjoint FineWeb-Edu and OpenAssistant source ranges, records revisions and selected IDs, writes manifests and source digests, and retries rate-limited API requests.
- The native trainer already publishes V3 checkpoints with tokenizer identity, dataset identity, parent checkpoint hash, session ID, optimizer state, and reload verification.
- The existing workflow already trains one level session, then stops at `AWAITING_HUMAN_VALIDATION` rather than automatically advancing.
- The current source selection policy is suitable as the acquisition foundation: FineWeb-Edu for English prose and OpenAssistant for supervised response-format practice, with held-out FineWeb rows separate from training.

## Gaps that block the approved Module 1 workflow

### 1. Curriculum granularity

The current state machine advances by broad level only. It does not represent Module 1 submodules or maintain a submodule pointer. A human PASS at Level 0 currently moves directly to Level 1, which is not the approved Module 1 path.

Required change: add explicit module and submodule identity, beginning with `module-1/submodule-1.1`, and advance exactly one submodule per invocation.

### 2. Module 1 data contract

The current preparer selects generic English rows but does not record a Module 1 submodule contract describing the intended symbol, whitespace, punctuation, and common-word coverage.

Required change: write a module/submodule contract and coverage summary into each session manifest and mastery packet. The initial dataset must be large enough to provide repeated examples of punctuation, whitespace, contractions, quoted text, short words, long words, digits, and common English function words while preserving disjoint validation and test splits.

### 3. Inference inspection

The current `cpp/tools` target graph has no dedicated native checkpoint-inspection executable. The trainer exposes `NextTokenModel::next_logits(...)` and the tokenizer exposes encode/decode, but `run.sh` only publishes metrics and a human prompt. It does not produce actual continuations.

Required change: implement a native C++ deterministic greedy decoder that loads the exact checkpoint, accepts an unseen prompt file, emits prompt-by-prompt continuations, and records token validity, EOS behavior, repetition, whitespace, and punctuation diagnostics.

### 4. Human mastery packet

The current mastery packet uses one generic prompt for every level and asks for at least five prompts. It does not state the specific Module 1 submodule objective, required prompt categories, acceptance threshold, or rejection threshold.

Required change: generate a submodule-specific mastery packet with the exact unseen prompt categories and human observations required for that submodule.

### 5. Configuration and budget

The current defaults are 100 FineWeb rows, 40 validation rows, 40 test rows, 100 pretraining steps, and 50 SFT steps. This is appropriate as a pipeline smoke/pilot budget but is not a defensible Module 1 qualification budget for all symbol and boundary patterns.

Required change: define an explicit pilot budget and a qualification budget. The first run should remain bounded for Colab, but it must be clear whether the result is a pipeline signal or a competency attempt. A first approved recommendation is 500 FineWeb training rows, 100 validation rows, 100 held-out test rows, 250 supervised training rows, 50 supervised validation rows, 500 pretraining steps, and 250 SFT steps for `module-1/submodule-1.1`. A retry uses a fresh disjoint range and the same budget unless the retry contract explicitly increases it.

### 6. Submodule-specific state transitions

The current retry stride and level stride are based on level numbers and cannot express the 12-step Module 1 sequence. A PASS at 1.1 must advance to 1.2; a FAIL at 1.1 must retry 1.1 from a fresh chunk; two controlled failures must stop in diagnosis.

Required change: replace or extend the state representation with module, submodule, attempt, parent checkpoint, pending session, and submodule status. The old level fields may remain for compatibility but cannot control Module 1 progression.

### 7. Test count and sequencing

The current workflow runs engineering regressions before each session, but it does not distinguish the user-facing competency test from automatic trainer tests. It also does not guarantee that later Module 1 submodules are blocked until earlier submodules have human PASS records.

Required change: run exactly one submodule session per `bash run.sh`, publish the user-facing test packet, and authorize only the next submodule after a valid human PASS.

## Audit conclusion

The existing curriculum is a solid level-based acquisition and checkpoint foundation, but it is not yet compliant with the approved academic Module 1 path. The first implementation work must therefore focus on submodule state, submodule-specific data contracts, native inference inspection, and session packets before beginning a new competency run. No current checkpoint should be auto-promoted to Module 1 completion.
