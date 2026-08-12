# External Source Notes

The package uses the following externally verified sources.

## FineWeb reference

Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb

The dataset card identifies FineWeb as English web data from Common Crawl and lists the dataset license as ODC-By. FineWeb is not used by the current native downloader because its published files are Parquet and this package must remain Python-free; it remains a documented future corpus option.

## OpenAssistant OASST1

Source: https://huggingface.co/datasets/OpenAssistant/oasst1
Revision endpoint: https://huggingface.co/api/datasets/OpenAssistant/oasst1
Pinned revision observed during preparation: `fdf72ae0827c1cda404aff25b6603abec9e3399b`.

The dataset card identifies Apache-2.0 metadata and reports a default dataset with train and validation splits. The repository tree exposed a directly downloadable compressed JSONL export at `2023-04-12_oasst_ready.messages.jsonl.gz`. The package pins that revision and derives deterministic train/validation/test message streams from English assistant messages, excluding deleted and synthetic messages when the fields are present. Individual record/source terms still require review before redistribution.

## Wikimedia dump

Source: https://dumps.wikimedia.org/enwiki/latest/
License guide: https://dumps.wikimedia.org/legal.html

The current index exposed `enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2` as a 299,138,062-byte compressed shard. The full multistream file was listed at 26,668,484,995 bytes. The Wikimedia license guide states that original text is generally available under GFDL-1.3 and CC BY-SA 4.0, subject to project terms, exceptions, and possible unnoticed infringements. The package defaults to the first shard and records the actual downloaded SHA-256; for exact repeatability, users should replace the moving `latest` alias with a dated dump URL and preserve its hash.

## Claim boundary

These sources support a real-data engineering experiment only. They do not establish representativeness, universal rights clearance, factual correctness, broad language competence, production readiness, or safety certification. `training_authorized` remains false.
