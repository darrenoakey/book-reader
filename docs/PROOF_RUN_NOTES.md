# Native proof runs

`discover_batch` can make multiple real-primary calls and must not be run inline without a strict external timeout. Run multi-stage native proofs detached with an explicit bounded wrapper, then sleep and inspect the log. Use focused unit tests or a single bounded primary call for inline verification.

## Wide-bio delta v2: compact-density contract

Raw 4096 and 8192 output-cap runs (`done_reason: length`) were caused by the model returning long dialogue/action sentences
as "facts". Delta version 2 repairs the output density at the source without losing information:

- Prompt, schema and value limit (`VALUE_MAX` 100 chars) ask for the SHORTEST exact contiguous phrase that states an
  explicit trait. Categories gain `voice` (voice, manner of speech, personality) and `power` (abilities).
- Nothing is capped per subject or category: several hair/eye rows, every named relative, every power and alias stay expressible.
  Only the same literal trait slot is deduplicated (`norm_slot`: case/punctuation/leading article or possessive insensitive).
- Local, deterministic and purely syntactic validation turns non-compact rows into typed pending items (never claims):
  `dialogue_value` (quotation marks, `?`/`!`), `value_not_compact` (per-category word cap, sentence/clause punctuation),
  `clause_value` (subject pronouns, speech verbs, copulas/auxiliaries/negation for noun-phrase categories, or a phrase led by the
  character's own name), `category_incompatible_value` (a bare character name filed as a trait). No semantic classifier is used.
- Unknown subjects stay `ref: ambiguous` and become `ambiguous_subject` pending; an actor is never guessed.
- `DELTA_VERSION` is 2 and the plan carries `compact_contract_sha256`, so the plan fingerprint (and every run directory
  identity) changes. Prior v1 raw/journal directories are immutable, incompatible history: `adaptive --from-out` refuses them
  (`adaptive_parent_mismatch`) and a v1 `--out` is refused (`plan_mismatch`). Start v2 in a NEW output directory.
