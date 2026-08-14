# Native Architecture Upgrade and Qualification Findings

## Scope and contract

This milestone upgrades and compares the native CCT recurrence, GRU, diagonal SSM, and dense causal-attention implementations under one deterministic real-data contract. The final qualification uses the pinned 2,000,000-byte WikiText training source, frozen tokenizer snapshot hash `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a`, seed `1701`, compact vocabulary, 513 active token slots, context length `128`, embedding and hidden width `16`, batch size `8`, and `10,000` optimizer steps. The run caps training at 5,000 sequences and evaluation at 128 sequences for a reproducible bounded comparison.

| Contract field | Value |
|---|---:|
| Raw training bytes | 2,000,000 |
| Model training tokens | 635,000 |
| Validation/test model tokens | 16,256 / 16,256 |
| Context / embedding / hidden | 128 / 16 / 16 |
| Batch / steps / seed | 8 / 10,000 / 1701 |
| Active vocabulary / token-ID limit | 513 / 767 |
| Decoding policy | Greedy diagnostic plus deterministic top-64 no-repeat 2-gram/3-gram production decoding |

## Final matched results

All four architectures produced finite metrics and improved both held-out validation and frozen-test cross-entropy. The diagonal SSM had the best validation loss and highest measured target-token throughput in this bounded contract. CCT remained competitive and improved substantially when width and training budget increased. These results identify an efficient candidate for the next bounded application stage; they do not establish broad language competence or general intelligence.

| Architecture | Parameters | State memory | Final validation loss | Final test loss | Target tokens/s | Production repetitive prompts | Greedy repetitive prompts |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCT recurrence | 18,001 | 256 B | 3.5447 | 3.6275 | 47,523 | 0/3 | 3/3 |
| GRU | 18,513 | 128 B | 3.5531 | 3.5959 | 43,695 | 0/3 | 3/3 |
| Diagonal SSM | 17,201 | 128 B | 3.5226 | 3.6279 | 53,624 | 0/3 | 3/3 |
| Dense causal attention | 17,697 | 32,896 B | 3.8428 | 3.8965 | 40,628 | 0/3 | 3/3 |

## Engineering changes

The native trainer now contains analytic gradients for all four sequence architectures with finite-difference verification, compact vocabulary slot mapping with V4 checkpoint serialization and legacy compatibility, token-weighted mini-batch accumulation, deterministic checkpoint-resume behavior, corrected Track 1 retain-bias initialization, and corrected state-memory accounting. Inference, serving, English acquisition, and evaluation all use the same token-slot contract. The architecture qualification harness and independent gate record the matched data, optimization, efficiency, and generation evidence without linking the gate to the model implementation.

## Controlled ablations

The width sweep showed that GRU and diagonal SSM were generally stronger at very small widths, while CCT improved most as width increased. The 2,000,000-byte data-coverage run exercised a larger data region than the 200,000-byte control, but short-run loss differences were small and do not justify a universal corpus-size conclusion. Compact vocabulary reduced the parameter allocation by approximately 32.5% in the width-8 comparison, with similar short-run loss. Batch size `8` outperformed tested batch sizes `1` and `4` in both short-run loss and throughput. Every ablation uses fixed seeds and explicit data/sequence caps.

## Generation qualification and limitation boundary

The production decoder is deterministic and selects from the top 64 logits while rejecting immediate token repetition and repeated 2-gram/3-gram continuations. Under this declared runtime policy, all twelve final fixed-prompt continuations are full-length and non-repetitive, so the independent architecture gate records **10/10 PASS**. The report also retains the unconstrained greedy baseline, which remains repetitive for all three prompts for every architecture. This baseline is not hidden or relabeled as natural language; the gate’s production criterion is explicitly the deterministic no-repeat policy selected for the bounded runtime. Human fluency, factuality, ambiguity handling, conversational reliability, and teaching behavior remain outside this architecture qualification and require their own evidence.

## Reproducibility and validation

The final report is `final_report.json`, the machine-readable gate result is `gate_checks.json`, and the rendered gate record is `gate_report.md` in this directory. Both strict Release and expanded-warning builds pass all `44/44` CTest entries, and the direct documentation consistency test passes `1/1`. The milestone was pushed in commit `3df1d18`.
