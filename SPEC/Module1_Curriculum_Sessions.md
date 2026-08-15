# Module 1 Session Contract

## Purpose

Module 1 teaches the model the visible structure of English before sentence grammar and conversation. Each submodule is trained and judged independently. A run trains exactly one submodule, then stops for human competency validation.

## Ordered submodules

| ID | Name | Learning objective | Next submodule after PASS |
|---|---|---|---|
| `1.1` | Character and symbol awareness | Reproduce letters, digits, punctuation, and common symbols without invalid or degenerate output | `1.2` |
| `1.2` | Whitespace and word boundaries | Preserve spaces, word separation, punctuation spacing, contractions, and quoted boundaries | `1.3` |
| `1.3` | Common word patterns | Learn frequent English words, short/long word shapes, common affixes, repeated letters, and function-word patterns | `1.4` |
| `1.4` | Stable short continuation | Continue unseen short English fragments while combining symbol, boundary, and word-pattern knowledge without collapse | Module 2 / sentence formation |

## Session procedure

Each invocation of `bash run.sh` performs exactly one submodule session:

1. Read the persisted Module 1 state from the Google Drive curriculum root.
2. Select the current submodule and attempt number.
3. Select a fresh disjoint FineWeb-Edu chunk and a fresh disjoint OpenAssistant chunk.
4. Train the native CCT model with pretraining followed by supervised fine-tuning.
5. Save the immutable session directory, checkpoint lineage, metrics, manifest, and resource report.
6. Run the native deterministic checkpoint inspector on the submodule’s unseen prompt packet.
7. Write a submodule-specific mastery packet containing the actual generated outputs and acceptance criteria.
8. Stop at `AWAITING_HUMAN_VALIDATION`.
9. A human PASS advances one submodule. A human FAIL retries the same submodule on a new disjoint chunk. Two valid failures open diagnosis and block progression.

## Shared model configuration

The default Module 1 configuration is deliberately compact but larger than the earlier 16-wide pilot:

- Architecture: native `Track1CctRecurrence`.
- Compact vocabulary: enabled.
- Embedding dimension: 32.
- Hidden dimension: 32.
- Context length: 128 tokens.
- Batch size: 16.
- Seed: 1701.
- Pretraining learning rate: the native trainer default curriculum rate.
- SFT learning rate: the native trainer default curriculum rate.
- Weight decay, gradient clipping, finite checks, and checkpoint lineage: enabled by the native trainer.

The model width remains fixed across Module 1. The curriculum changes the data chunk and learning budget, not the inference architecture.

## Submodule budgets

The budgets below are qualification budgets, not claims that the model must pass automatically. They are chosen to provide repeated coverage while remaining bounded for Colab. Each retry uses the same budget and a new source range unless a new budget is explicitly approved after diagnosis.

| Submodule | FineWeb train / validation / held-out test | OpenAssistant SFT train / validation | Pretraining / SFT steps | Human prompts |
|---|---:|---:|---:|---:|
| `1.1` | 500 / 100 / 100 rows | 250 / 50 rows | 500 / 250 | 10 |
| `1.2` | 750 / 150 / 150 rows | 375 / 75 rows | 750 / 375 | 10 |
| `1.3` | 1,000 / 200 / 200 rows | 500 / 100 rows | 1,000 / 500 | 10 |
| `1.4` | 1,250 / 250 / 250 rows | 625 / 125 rows | 1,250 / 625 | 12 |

The first run is `1.1` only. The runner must not allocate or train `1.2`, `1.3`, or `1.4` in the same invocation.

## Data selection policy

- FineWeb-Edu supplies English educational prose for the language-model objective.
- OpenAssistant supplies English-filtered assistant and user text for supervised response-format practice.
- Training, validation, held-out test, and human prompt material are disjoint by stable source identity where source identities exist.
- Every session records the source revision, selected IDs, offsets, filters, row counts, text hashes, and submodule identity.
- A retry changes the source range while preserving the submodule objective and model configuration.
- The human prompt packet is never used as a training file.

## Submodule learning contracts

### `1.1` Character and symbol awareness

The model should learn:

- Uppercase and lowercase letters.
- Digits.
- Periods, commas, question marks, exclamation marks, colons, and semicolons.
- Apostrophes, quotation marks, parentheses, and hyphens.
- Symbols at the beginning, middle, and end of text.
- Short continuations that do not become empty, invalid, or endlessly repetitive.

Human acceptance requires at least 8 of 10 unseen prompts to produce inspectable non-degenerate output, with both adversarial symbol cases valid. No PASS is allowed for invalid output, all-EOS output, empty output, or systematic repeated-symbol collapse.

### `1.2` Whitespace and word boundaries

The model should learn:

- Spaces between words.
- Spaces after punctuation.
- Word boundaries around quotation marks and parentheses.
- Contractions such as `don't`, `can't`, and `we're`.
- Word beginnings and endings.
- Punctuation adjacent to words without uncontrolled spacing.

Human acceptance requires at least 8 of 10 prompts to preserve readable word separation and punctuation boundaries, with no systematic whitespace collapse or character-level loops.

### `1.3` Common word patterns

The model should learn:

- Frequent function words such as `the`, `a`, `is`, `of`, `to`, and `and`.
- Common short and long word shapes.
- Repeated letters.
- Common prefixes and suffixes.
- Singular and plural word forms.
- Frequent educational and everyday vocabulary patterns.

Human acceptance requires at least 8 of 10 unseen prompts to continue with recognizable English word patterns without unrelated or degenerate output.

### `1.4` Stable short continuation

The model should combine the earlier subskills when continuing short unseen English fragments. It should preserve symbols, spaces, word boundaries, common word patterns, and punctuation at the same time.

Human acceptance requires at least 10 of 12 unseen prompts to remain valid, readable, non-degenerate, and structurally consistent with the prompt. PASS authorizes the next academic module: sentence formation and local grammar.

## Verdict meanings

- `PASS`: the declared submodule is demonstrated on the required unseen prompts; the next invocation may train the next submodule.
- `FAIL`: the declared submodule is not demonstrated; the next invocation retries the same submodule on a fresh disjoint chunk.
- `INSUFFICIENT_EVIDENCE`: the checkpoint or generated outputs could not be inspected; the inference harness must be repaired before a competency verdict.
- `ARCHITECTURE_DIAGNOSIS_REQUIRED`: the same submodule failed twice under valid data, identity, and evaluation controls.
