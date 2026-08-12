# WikiText direct-file provenance

The pinned Salesforce/wikitext revision `b08601e04326c79dfdd32d625aee71d232d685c3` exposes `wikitext-2-raw-v1/{train,validation,test}-00000-of-00001.parquet` as direct files. The repository card lists CC BY-SA 3.0 and GFDL metadata. The native Track 1 implementation currently consumes the Hugging Face rows endpoint for WikiText, and bounded remote pagination encountered HTTP 429 at validation offset 2600. The direct Parquet files are not yet wired because the repository requires a native C++20 parser and has no Parquet dependency.

The SQuAD acquisition is already migrated to the pinned GEM flat JSON files, which are directly parsed natively.
