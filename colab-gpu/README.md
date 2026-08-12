# CCT Native C++20/CUDA Colab GPU Package

This directory provides a **strict native C++20/CUDA** training and evaluation workflow. It does not install Python, PyTorch, Transformers, or a Python data loader. The package downloads real public data, compiles a native dataset preparer and CUDA recurrent trainer, creates deterministic token-stream train/validation/test files, runs a bounded large-corpus pretraining stage, continues with an OASST1 supervised message-continuation stage, checkpoints both stages, and writes machine-readable metrics.

## One-command execution

Open a Colab notebook, select **Runtime → Change runtime type → T4 GPU** or another CUDA-enabled runtime, clone the repository, and run:

```bash
cd CCT
bash run.sh
```

The wrapper delegates to `colab-gpu/run.sh`. The script fails closed if `nvcc`, `nvidia-smi`, `curl`, `bzip2`, `gzip`, `g++`, or `sha256sum` is unavailable, or if no CUDA device is visible.

## What the command downloads

The default large-corpus input is the first current English Wikimedia multistream XML shard, approximately 299 MB compressed at the time this package was prepared [1]. It is a real data shard, not a synthetic fixture. The parser extracts article text from the XML stream, applies a minimum-length filter, emits byte-fallback token IDs, inserts sequence boundaries, and assigns deterministic 90/5/5 train/validation/test splits using a stable document ordinal hash.

The supervised input is the pinned OpenAssistant OASST1 ready-message JSONL export at revision `fdf72ae0827c1cda404aff25b6603abec9e3399b` [2]. The native parser selects English assistant messages, excludes deleted and synthetic messages when those fields are present, prepends an assistant control marker, and assigns deterministic 80/10/10 train/validation/test splits using the stable message ID hash. This is a language-model continuation objective over assistant messages; it is not a claim that the package reconstructs the full OASST conversation tree.

Wikimedia original text is generally available under GFDL and CC BY-SA 4.0 subject to the Wikimedia licensing information [3], project terms, and exceptions. OASST1 metadata identifies the dataset repository as Apache-2.0; inspect the dataset card [2] and individual record/source terms before redistribution. `run.sh` records the actual downloaded byte hashes and URLs in each run artifact. The Wikimedia `latest` alias is moving; for an exactly repeatable experiment, replace `WIKI_URL` with a dated dump URL and preserve its recorded hash.

## Native execution stages

The workflow first compiles `colab-gpu/native/prepare.cpp` with `g++ -std=c++20 -Wall -Wextra -Wpedantic -Werror`. It then compiles `colab-gpu/native/cuda_train.cu` with `nvcc -std=c++20 -O3`. The CUDA trainer uses a bounded CCT-family recurrent token model with byte-fallback vocabulary IDs, GPU forward/backpropagation, clipped Adam-style updates, validation/test scoring, and atomic checkpoint replacement. It records GPU name, compute capability, token counts, losses, perplexity, accuracy, dataset identity, tokenizer identity, and checkpoint paths.

The first stage trains on the Wikimedia token stream and writes `checkpoints/cct_base_cuda.bin`. The second stage loads the base weights, resets optimizer moments for the new SFT data identity, trains on OASST1 assistant messages, and writes `checkpoints/cct_oasst_sft_cuda.bin`. Each stage evaluates validation and test streams. Checkpoints contain the model configuration, optimizer state, step, corpus identity, tokenizer identity, and parameter arrays. Ordinary interruption resume requires exact dataset/configuration identity; base-to-SFT transfer explicitly permits the new SFT dataset identity while requiring the tokenizer and model configuration to match.

## Resource and run controls

The defaults are deliberately bounded for a Colab GPU and can be changed with environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `PRETRAIN_STEPS` | 1000 | Native CUDA optimizer steps on Wikimedia |
| `SFT_STEPS` | 600 | Native CUDA optimizer steps on OASST1 |
| `MAX_TRAIN_TOKENS` | 16,000,000 | Maximum train tokens loaded per stage |
| `MAX_VALIDATION_TOKENS` | 2,000,000 | Maximum validation tokens loaded per stage |
| `MAX_TEST_TOKENS` | 2,000,000 | Maximum test tokens loaded per stage |
| `BATCH_SIZE` | 32 | Sequences per CUDA update |
| `CONTEXT_LENGTH` | 64 | Recurrent context window |
| `HIDDEN_DIM` | 32 | CCT recurrent hidden width |
| `EMBEDDING_DIM` | 32 | Token embedding width |
| `CHECKPOINT_EVERY` | 100 | Optimizer steps between atomic checkpoints |

For a longer bounded run:

```bash
MAX_TRAIN_TOKENS=64000000 PRETRAIN_STEPS=5000 SFT_STEPS=2500 bash run.sh
```

The full current Wikimedia shard can be used by setting larger limits, but disk, host RAM, download time, and GPU time must be checked first. A single Colab session is not a substitute for distributed pretraining infrastructure.

## Output artifacts

Each run creates a timestamped directory under `colab-gpu/artifacts/`. It contains the source manifest, archive SHA-256 records, prepared split manifests, run configuration, per-step logs, pretraining metrics, SFT metrics, and a release record. A successful run is not automatically a production authorization. The release record deliberately sets `training_authorized` to `false` and states that independent evaluation and human review remain required.

## Acceptance checks

A valid run must show `status: PASS` for both native CUDA metric records, positive finite validation/test metrics, nonzero train/validation/test token counts, checkpoint files for both stages, matching tokenizer identity, and no `cuda_train error` output. An interrupted run may be resumed with the exact same command arguments plus `--resume` if the user invokes the native trainer directly; the wrapper itself starts a fresh stage only when the expected checkpoint is absent. Corrupt or mismatched checkpoints are rejected rather than silently reused.

## Claim boundary

This package is an engineering path to obtain **real GPU measurements on a bounded native CCT-family model**. It does not prove broad language competence, production-scale training, factuality, human-level instruction following, safety certification, CUDA performance portability, multi-GPU scaling, or superintelligence. The current repository’s formal Stage 13 release remains the governing checkpoint; a Colab run is a separate experiment and must be reviewed before being treated as evidence for a later roadmap stage.

## References

[1]: https://dumps.wikimedia.org/enwiki/latest/ — Wikimedia English Wikipedia dump index.

[2]: https://huggingface.co/datasets/OpenAssistant/oasst1 — OpenAssistant OASST1 dataset card and revision metadata.

[3]: https://dumps.wikimedia.org/legal.html — Wikimedia dump licensing information.
