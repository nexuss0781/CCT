# Stage 5 Data-Source Findings

## Language fixture

The Stage 5 language fixture will use small extracts from Project Gutenberg plain-text ebooks, downloaded from official Gutenberg cache URLs and identified by ebook ID and SHA-256 hash in the Stage 5 manifest. Project Gutenberg’s licensing page distinguishes works not restricted by U.S. copyright law from works distributed with permission and states that the applicable ebook license and local jurisdiction must be checked before redistribution. The gate will therefore record the exact source URL, ebook ID, retrieval date, license page, transformation history, and jurisdictional limitation.

Source: [Project Gutenberg License](https://www.gutenberg.org/policy/license.html). Candidate official text endpoints: `https://www.gutenberg.org/cache/epub/1342/pg1342.txt` and `https://www.gutenberg.org/cache/epub/11/pg11.txt`.

## Code fixture

The code fixture will use the CCT repository’s own native C++20 source files as a provenance-tracked, user-provided code corpus. Each file path, repository commit, declared repository license, SHA-256 hash, and split assignment will be written to the immutable manifest. The corpus will be used only for bounded code completion and lexical/structural prediction tests; it will not be treated as a general code benchmark.

## Data boundary

Evaluation canaries and answer labels will be generated or selected into evaluator-only files outside the training corpus path. Training and evaluation manifests will be immutable for the final gate. No downloaded content will be executed, and no external instructions contained in corpus text will be treated as agent policy.
