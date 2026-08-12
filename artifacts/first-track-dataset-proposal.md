# CCT-ASE First Small Training Track

## Decision

For the first architecture-maturation track, use **WikiText-2 for pretraining** and **SQuAD 2.0 for supervised fine-tuning and final held-out testing**. Do not use the full Wikipedia shard or OASST1 in this first track. OASST1 is retained as the next instruction-quality track after the architecture passes this compact end-to-end loop.

This choice gives the first track a small natural-language corpus, a task-specific supervised objective, an explicit abstention test, fixed source splits, and a clean boundary between training data and final evaluation data.

## Dataset roles and exact first-track budgets

| Role | Source | First-track selection | Use |
|---|---|---:|---|
| Language-model pretraining | Salesforce WikiText-2 | Official `train` configuration, capped at 2,000,000 tokens or the complete available train stream if smaller | Update model parameters and measure next-token learning |
| Language-model validation | Salesforce WikiText-2 | Official `validation` configuration, never used for updates | Select checkpoint and measure loss/perplexity |
| Language-model test | Salesforce WikiText-2 | Official `test` configuration, read only after training is frozen | Final next-token test loss/perplexity |
| Supervised fine-tuning | SQuAD 2.0 training set | 8,000 examples: 4,000 answerable and 4,000 unanswerable, selected by stable question-ID hash; train/eval split 7,200/800 | Train question-context answering and abstention behavior |
| Fine-tuning evaluation | SQuAD 2.0 training subset | The held-out 800 examples, selected before optimization and excluded from all updates | Checkpoint selection and early failure detection |
| Final task test | SQuAD 2.0 official dev set | Entire official dev set, frozen and evaluated once after model selection | Report exact match, token F1, answerability AUROC/AUPRC, abstention precision/recall, and unsupported-answer rate |

The pretraining and fine-tuning phases remain separate. The SQuAD final dev set is not used for model selection. The WikiText-2 test set is not used for any training decision.

## Why WikiText-2 first

The official WikiText card describes the corpus as text extracted from verified Good and Featured Wikipedia articles and publishes the WikiText-2 train, validation, and test configurations. The card reports 36,718 train rows, 3,760 validation rows, and 4,358 test rows for the raw and processed WikiText-2 configurations. The pinned repository metadata lists CC BY-SA 3.0 and GFDL licensing, so the repository must retain attribution and license notices with the manifest. [1]

WikiText-2 is preferable to the existing full Wikipedia shard for this first loop because its fixed small splits make repeated native CPU/GPU runs practical and make regressions interpretable. It is natural text rather than synthetic stories, while still being small enough for architecture iteration.

## Why SQuAD 2.0 for fine-tuning and testing

The official SQuAD site states that SQuAD 2.0 combines answerable questions with more than 50,000 adversarially written unanswerable questions and explicitly tests both reading comprehension and abstention. The site provides the CC BY-SA 4.0 license, a training set, a dev set, an evaluation script, and a sample prediction file. [2]

SQuAD 2.0 gives the first track one consistent supervised task: consume a context and question, return an answer span or abstain. That directly exercises CCT context handling, structured output, evidence grounding, and refusal behavior without requiring a larger instruction mixture. The official dev set remains a true final holdout.

## Acquisition route and pinned mirror provenance

The native preparer retains `Salesforce/wikitext` as the declared WikiText source identity while acquiring the three fixed raw split members from the directly downloadable `wikitext-2-raw-v1.zip` archive in `ggml-org/ci`, pinned to revision `927b3642933080f1b0e811e2f916e14c292992f9`. The archive is cached once, extracted natively, checked against the expected 36,718/3,760/4,358 split sizes, and recorded through member-level SHA-256 digests in the manifest. This avoids large-scale pagination against the Hugging Face rows endpoint. [6]

The native preparer retains `rajpurkar/squad_v2` as the declared upstream benchmark identity while acquiring SQuAD records from the directly downloadable `GEM/squad_v2` flat JSON files. The mirror is pinned to revision `67199807729e631955056c71c258b7acbee548a3`, is labeled CC BY-SA 4.0, and preserves the original SQuAD question IDs, contexts, answer text, and codepoint answer offsets. It exposes `gem_data_split/train.json` for governed SFT selection and `gem_data_split/validation.json` for the frozen final test. [5]

This direct-file route replaces large-scale rows-endpoint pagination for SQuAD because it is a single cacheable artifact per split and eliminates the observed HTTP 429 failure mode. The manifest records the GEM dataset ID, immutable revision, direct URL, raw-file digest, acquisition type, and `rajpurkar/squad_v2` upstream identity. The parser converts the retained codepoint offsets to UTF-8 byte offsets and fails closed if an answer span does not exactly match its context.

## Why OASST1 is deferred

OASST1 is a strong later fine-tuning source. Its authoritative dataset card lists Apache-2.0 metadata, 84.4k training rows, 4.4k validation rows, conversation-tree identifiers, parent/message identifiers, language, rank, review counts, and quality labels. [3] However, its multilingual conversation trees, ranking fields, annotation filtering, and broader assistant distribution add a second data-governance problem before the architecture has passed this small, task-consistent loop. It should be the next instruction-quality track, not mixed into the first architecture test.

## Governance and reproducibility contract

The repository will pin the exact source URLs, dataset revisions, license metadata, download hashes, file hashes, parser version, tokenizer snapshot, and selection seed. Every selected SQuAD record retains its original question ID, article/title identity, answerability label, context digest, and source provenance. A question ID or article identity may occur in only one of train, fine-tuning evaluation, or final test.

The native preparer will fail closed on malformed JSON, duplicate IDs, missing contexts, missing answerability labels, overlapping IDs, unsupported license metadata, and manifest/hash mismatches. The output manifests will record the exact selected IDs, split counts, token counts, token caps, and content digests.

Every run will use the same tokenizer snapshot, context length, batch contract, optimizer schedule, checkpoint format, evaluation code, and three declared random seeds. The report will include the CCT model, matched native recurrent control, parameter count, wall-clock time, peak memory, train/validation/test losses, exact-match/F1/abstention metrics, checkpoint identity, and failure counts.

## First-track pass criteria

The first track passes only if the native preparer produces deterministic manifests twice, all stream validators pass, pretraining validation loss improves over the frozen initialization baseline, the final WikiText-2 test stream is evaluated only after checkpoint selection, SQuAD fine-tuning improves held-out evaluation over the pretraining-only checkpoint, and the final SQuAD dev report is complete with answerable and unanswerable slices. Any overlap, malformed source, missing provenance, non-finite metric, or unauthorized use of the final test set fails the track.

## Source references

[1]: https://huggingface.co/datasets/Salesforce/wikitext "Salesforce WikiText dataset card"
[2]: https://rajpurkar.github.io/SQuAD-explorer/ "Official Stanford SQuAD 2.0 site"
[3]: https://huggingface.co/datasets/OpenAssistant/oasst1 "OpenAssistant OASST1 dataset card"
[4]: https://huggingface.co/datasets/roneneldan/TinyStories "TinyStories dataset card, evaluated and deferred for this track"
[5]: https://huggingface.co/datasets/GEM/squad_v2/tree/67199807729e631955056c71c258b7acbee548a3 "GEM SQuAD v2 pinned repository files and dataset card"
[6]: https://huggingface.co/datasets/ggml-org/ci/tree/927b3642933080f1b0e811e2f916e14c292992f9 "Pinned WikiText raw archive mirror"
