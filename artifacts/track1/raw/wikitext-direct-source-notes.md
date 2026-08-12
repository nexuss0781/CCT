# WikiText direct acquisition provenance

The canonical Track 1 dataset identity remains `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, pinned to revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Its repository metadata declares CC BY-SA 3.0 and GFDL. The Hugging Face rows endpoint was retained as legacy provenance but is not used for production acquisition because it can return HTTP 429 while paginating thousands of rows.

The native C++20 preparer instead downloads one direct raw archive from the pinned Hugging Face mirror `ggml-org/ci` revision `927b3642933080f1b0e811e2f916e14c292992f9`:

```text
https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip?download=true
```

The downloaded archive SHA-256 is `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`. Its direct members are `wikitext-2-raw/wiki.train.raw` (36,718 lines), `wikitext-2-raw/wiki.valid.raw` (3,760 lines), and `wikitext-2-raw/wiki.test.raw` (4,358 lines), matching the declared fixed split counts. The preparer caches the archive once, extracts each member atomically, records a per-member SHA-256 digest in the manifest, and fails closed on missing archive members or short splits.

A full production preparation completed with this path using five direct source artifacts: one WikiText archive and the two direct GEM SQuAD JSON files, with the archive subsequently reused for the remaining WikiText split members.
