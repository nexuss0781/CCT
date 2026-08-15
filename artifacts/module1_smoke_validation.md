# Module 1 Smoke Validation Evidence

## Execution

The command used an isolated temporary root with `SMOKE=1`, `CURRICULUM_MODE=1`, and `CURRICULUM_MODULE1=1`. It completed one session only: `module-1-submodule-1-attempt-0`.

## Engineering results

- Native strict build completed for the preparation, session, inspector, trainer-regression, Track 1, and documentation targets.
- NLP trainer regressions: 14/14 passed.
- Dataset preparation tests: 9/9 passed.
- Documentation consistency: 1/1 passed.
- Preparation published 4/2/2 FineWeb rows and 4/2 OpenAssistant rows under an isolated Module 1 data directory.
- Native training published a reloadable child checkpoint from `GENESIS`.
- Native deterministic inspection published 10 prompt outputs in `inference.jsonl`.
- The inspector reported 10 prompts and 8 token-valid/UTF-8-valid outputs.

## Behavioral evidence

The bounded smoke model produced repeated `r` characters, repeated `]` characters, repeated tails, missing punctuation, and invalid UTF-8 byte sequences on multiple prompts. Examples included a maximum same-token run of 64 and repeated-tail diagnostics. This is expected evidence that the smoke configuration is not a language-quality result; it proves that the new inspector exposes the failure behavior instead of hiding it behind token loss.

## Conclusion

The Module 1 orchestration and inspection path is operational under the smoke contract. The smoke output must not receive a human competency PASS. Qualification requires the declared Module 1.1 budget and human review of the actual persisted inference packet.
