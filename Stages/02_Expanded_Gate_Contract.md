# Stage 2 Expanded Gate Contract

**Purpose.** This contract closes the limitations recorded in the first Stage 2 gate. Stage 3 remains prohibited until every mandatory check below is green.

| Limitation | Required implementation | Mandatory evidence | Pass threshold |
|---|---|---|---|
| Complex state | Native complex-valued recurrent state with explicit real/imaginary update convention, finite-value checks, streaming path, scan equivalence, and checkpointing | Complex unit tests and gate artifact | Reference/scan/step max error `<1e-12`; no non-finite values |
| Normalization | Configurable state RMS and output RMS normalization, disabled by default, with exact on/off ablation | Same-seed normalized vs unnormalized metrics and state/output norms | Enabled path finite; ablation changes only declared normalization behavior; checkpoint preserves flags |
| Segmented mask scan | Prefix scan over contiguous active segments while masked positions preserve state and emit read-only outputs | Masked loop/scan/chunk equivalence across multiple segment boundaries | Max output/state error `<1e-12` |
| Trained matched baselines | Trainable, matched micro-baselines: dense causal attention, GRU, and diagonal SSM, plus CCT | Same task generator, parameter report, optimizer budget, training/evaluation metrics, and raw timings | All models train deterministically; each report includes loss, parameter count, memory, and timing |
| Algorithmic coverage narrow | Add associative recall, parity/state tracking, and selective overwrite to copy/delayed recall | Per-task train and held-out length tables | Each task meets its declared minimum; no teacher-forcing-only pass |
| Ablations incomplete | Measure gate removal, complex vs real, MIMO width, normalization, and metadata-channel contribution | JSON/Markdown ablation matrix | Every row has a reproducible result and interpretation |
| Efficiency evidence limited | Compare scan/loop and all baselines across declared lengths with raw timing and memory data | Scaling table and slope fit | No hidden quadratic allocation; recurrent decode memory is constant in sequence length |

## Stop conditions

Any NaN, infinity, silent mask-state mutation, checkpoint mismatch, nondeterministic rerun, missing baseline metric, or unreported failed task is an immediate **FAIL**. A green build without complete evidence is not a pass.

## Transition rule

The strengthened Stage 2 gate reports `PASS` only when all **12 mandatory checks** and every limitation-closure metric are green. `PASS` authorizes Stage 3 preparation only; Stage 3 implementation requires explicit user approval.
