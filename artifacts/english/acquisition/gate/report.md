# L1 English Acquisition Gate

**Status:** `PASS`
**Checks:** 14/14

- `PASS` **status_pass** — runner status is PASS
- `PASS` **native_backend** — native C++ backend is recorded
- `PASS` **source_identity** — source, tokenizer, WikiText, and CoLA identities are present
- `PASS` **full_blimp_coverage** — all 67 BLiMP files and at least 6,700 pairs are scored
- `PASS` **blimp_finite_above_chance** — trained BLiMP accuracy is finite and at least chance
- `PASS` **blimp_beats_control** — trained BLiMP preference count beats the matched no-training control
- `PASS` **cola_beats_control** — adapted CoLA preference beats control and remains above chance
- `PASS` **validation_loss_improves** — held-out WikiText validation loss improves
- `PASS` **frozen_test_improves** — frozen WikiText test loss improves
- `PASS` **checkpoint_hash** — checkpoint SHA-256 matches report identity
- `PASS` **checkpoint_nonempty** — checkpoint is durable and non-empty
- `PASS` **side_effect_isolation** — external actions are disabled
- `PASS` **evaluation_only_boundary** — final report was produced through evaluation-only scoring
- `PASS` **generation_validity** — bounded generation outputs are non-empty and valid UTF-8
