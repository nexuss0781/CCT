# English Acquisition Source Notes

The current English acquisition contract uses the following externally verified sources.

| Source | Verified facts used by the contract | URL |
|---|---|---|
| Salesforce WikiText dataset card | WikiText is an English language-modeling corpus. The `wikitext-103-raw-v1` subset has fixed train, validation, and test splits with approximately 1.81M, 3.76k, and 4.36k rows, respectively; the raw download is approximately 191.98 MB and the dataset is distributed under CC BY-SA/GFDL metadata. | https://huggingface.co/datasets/Salesforce/wikitext |
| BLiMP repository | BLiMP contains 67 English minimal-pair JSONL datasets, each with 1,000 pairs, and fields including `sentence_good`, `sentence_bad`, `field`, `linguistics_term`, and `pairID`. The repository identifies the benchmark as CC-BY. | https://github.com/alexwarstadt/blimp |
| BLiMP paper | The benchmark evaluates whether a language model assigns higher probability to the acceptable sentence; it reports 67 minimal-pair datasets and aggregate human agreement of 96.4%. | https://aclanthology.org/2020.tacl-1.25/ |
| TinyStories paper | TinyStories is a synthetic English story corpus designed for small-model coherence experiments. It is retained as a possible later controlled fluency corpus, not as the primary real-English source for this milestone. | https://arxiv.org/abs/2305.07759 |
| CoLA / GLUE source | The official CoLA archive contains `train.tsv` (8,551 rows) and `dev.tsv` (1,042 rows) with sentence acceptability labels. The archive was acquired from the official GLUE file endpoint and hashed locally as `f212fcd832b8f7b435fb991f101abf89f96b933ab400603bf198960dfc32cbff`. It is used only as a grammar-adaptation training/development source, never as BLiMP evaluation. | https://dl.fbaipublicfiles.com/glue/data/CoLA.zip |

The locally acquired BLiMP repository archive is `artifacts/english/raw/BLiMP.zip` with SHA-256 `8c043d41701fff01cabf22fb9a2fd8e15cecac5240b777ff41680f4eb7fffc17`; it contains exactly 67 JSONL files. The current pinned WikiText Track 1 preparation remains WikiText-2; the next broader run must not describe that source as WikiText-103 unless a separately pinned WikiText-103 source is acquired and parsed natively.
