# Focused-English Continual-Learning Dataset Research

## Verified source findings

FineWeb is an English web corpus released through Hugging Face under ODC-By metadata. Its dataset page reports more than 18.5 trillion tokens in the current presentation, many Common Crawl subsets, English language metadata, per-document identifiers, URLs, dates, language scores, and token counts. The page also exposes smaller sample subsets, which makes it suitable for deterministic chunk selection rather than requiring an all-at-once download.

Dolma is an English-only, 3-trillion-token corpus from Ai2 combining web content, academic publications, code, books, and encyclopedic materials. Ai2 documents language identification, quality filtering, deduplication, PII masking, toxicity filtering, source diversity, and decontamination. Dolma is distributed under the Ai2 ImpACT license with access and derivative obligations, so it is a strong research candidate but requires explicit license handling in the manifest.

OpenAssistant/oasst1 is an Apache-2.0 dataset with approximately 88.8k rows, English plus other languages, train and validation splits, conversation-tree metadata, human review fields, ranking information, and toxicity/quality annotations. It is suitable as an instruction/SFT candidate after a native English-only filter and deterministic tree-to-dialogue conversion, but it is not a general pretraining corpus.

GLUE is an English evaluation benchmark of nine sentence or sentence-pair tasks plus a diagnostic set. It is suitable as a held-out competency-evaluation family, not as the main pretraining corpus. Its task diversity supports staged tests for acceptability, sentiment, similarity, entailment, paraphrase, and inference.

## Initial selection direction

Use a smaller reproducible FineWeb English subset or existing governed WikiText source for the first native curriculum sessions, because the workflow must select explicit document ranges, record source identities, and avoid downloading a multi-trillion-token corpus. Use English-filtered OpenAssistant conversation trees for supervised dialogue-format practice, with SQuAD-style answer-target examples retained for grounded comprehension. Keep GLUE/SuperGLUE-like tasks outside training and reserve them for competency validation. Consider Dolma only as a later scale-up source after license and access controls are implemented.

## Design constraints

Every source must have a pinned URL or dataset revision, license record, source digest, language-filter rule, document-range selection, split ownership, and contamination exclusion record. Training chunks and validation/test chunks must be disjoint by stable document identifier. Human mastery validation must be external to the automatic loss gate; automatic metrics may report evidence but cannot mark a competency mastered on their own.

## FineWeb-Edu follow-up

The official FineWeb-Edu dataset page describes `sample-10BT` as an approximately 10-billion-GPT2-token random subset, with larger `sample-100BT` and full-scale variants. The repository exposes Parquet files, and the sample-10BT directory is approximately 28.5 GB. This makes FineWeb-Edu a strong quality-oriented pretraining source, but a native C++ ingestion path must either add a Parquet reader or use an official row/API or converted text endpoint; the workflow must not silently depend on Python. For the first curriculum implementation, a smaller governed text-compatible source remains operationally safer, while FineWeb-Edu can be adopted once the native ingestion contract is implemented and tested.

## Native acquisition feasibility

The Hugging Face datasets-server rows endpoint responds successfully for `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, split `train`, and exposes text, stable id, URL, language, language score, token count, and educational score fields. It also responds for `OpenAssistant/oasst1`, configuration `default`, split `train`, and exposes message id, parent id, text, role, language, review metadata, deletion state, rank, tree id, and quality/toxicity labels. These APIs are viable for a native C++ downloader that requests fixed offset/length ranges, applies English and quality filters, and records exact source rows and digests.
