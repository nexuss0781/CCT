# Stage 11 Trainable Native NLP Core Gate Report

**Status:** `PASS`  
**Selected model:** `track1_cct_recurrence`  
**Tokenizer hash:** `902e5a44f372a3d972b6f21036d62d7878f1d6907805c841e49aa84297ba7b0a`  
**Checkpoint hash:** `fb15d6e0900d047b28bc38662e826861f16a423e5ce896d68891bceacd3f66da`

## Evidence boundary

This gate exercises a real categorical next-token objective over bounded slices of governed real text and native C++ sources, application-shaped code/JSON/Unicode/separator fixtures, a held-out validation slice, three Track 1 recurrence seeds, matched native controls, analytic/finite-difference gradients, optimizer schedules, checkpoint interruption/resume, cursor identity, contamination rejection, and fail-closed invalid inputs.

## Claim boundary

Stage 11 is a small controlled CPU training pilot. It does not establish broad language competence, scale efficiency, factuality, safety, instruction following, retrieval grounding, production usefulness, or general intelligence. `training_authorized` remains false and Stage 12 requires explicit approval.
