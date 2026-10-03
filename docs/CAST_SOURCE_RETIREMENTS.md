# Caretaker source-reviewed retirement contract (`retire_actor`)

A **prepared prose actor** (registry `origin: "prepared"`, discovered from the book rather than part of the original
cast) leaves the cast preparation registry **only** through this input. Nothing is inferred or automatic: an actor the
file does not name is never touched, and the producer never retires an actor on its own.

Caretaker writes one complete `cast_source_reviewed_retirements.json` snapshot by atomic rename, beside the project's
other cast files. The producer reads it at a cast batch boundary, right after the root approvals file.

```json
{
  "contract": "cast_source_reviewed_retirement",
  "version": 1,
  "source_sha256": "<full source book sha256>",
  "registry_sha256": "<registry_digest(progress['registry']) of the registry the caretaker reviewed>",
  "retirements": [{
    "action": "retire_actor",
    "actor_id": "the prepared actor's id",
    "reviewer_role": "caretaker",
    "factual_basis": "why the source shows this label is not a living identity (must not claim a human review)",
    "evidence": [
      {"chapter_file": "NN-part_NN.txt", "chapter_sha256": "...", "unit_id": "c00s00000",
       "unit_quote": "exact immutable unit", "unit_quote_sha256": "..."}
    ],
    "own_refs": [
      {"chapter": "NN-part_NN.txt", "chapter_sha256": "...", "unit_id": "c00s00000",
       "quote_sha256": "...", "label": "exact label", "span_start": 0}
    ]
  }]
}
```

`registry_sha256` is `src.cast_freeze.registry_digest(registry)`: sha256 of the stable JSON
(`sort_keys`, compact separators) of the **whole** `registry` object in `cast_preparation_progress.json` at the moment
the producer reads the file. Any batch that changes the registry changes the digest, so the caretaker snapshots it
against the current progress file.

`evidence` is one or more exact immutable units (the witness shape of the v2 context file); at least one literally names
the actor. `own_refs` is the **complete** set of the actor's own exact name references over the whole book (every
immutable name reference whose label normalizes to the actor id or name): no more, no fewer. Each becomes one
mention-scoped `non_character` record.

## What the producer guarantees

**Fail closed (whole file, nothing written)** on: wrong contract/version/fields, a different book sha, a registry digest
that is not the exact current one, any entry with a wrong schema/action, a malformed or repeated `actor_id`, any text
claiming a human review, or a source chapter that cannot be read as exact evidence.

**A bad entry blocks that actor alone**: a typed `source_retirement_blocked` pending row in stage
`cast_source_retirement` (which blocks cast freeze); nothing is written for it, the other entries still apply. An entry
is blocked when any of these hold:

1. the actor is `narrator`, an original anchor, not `origin: "prepared"`, absent from the registry, or has an entry in the
   original `characters.json` / `voices.json` (original anchors and profiles are never retired);
2. the reviewer role is not a trusted role (`caretaker`) or the factual basis is empty;
3. an evidence unit is not exact in the current source bytes, or none literally names the actor;
4. `own_refs` is not exactly the actor's complete own reference set in the current source (or the actor has none);
5. the actor is still referenced, so retiring it would drop an actor other things resolve to: another alias key maps to
   it, its id is itself an alias, a mention-scoped `alias` record targets it, the alias audit names it, or an open
   `cast`, `cast_context` or `cast_root_approval` pending row references it. A `cast_quality` flag on the actor is *not*
   a blocker: it is answered by the retirement (resolved with outcome `actor_retired_source_reviewed`);
6. one of its own references already carries a scoped `alias` decision.

Because a referenced actor is never retired, no speaker is left pointing at the narrator or at an inactive alias: after
retirement every alias in the registry still resolves to an active actor.

## What is written (only for an unblocked entry), in this order

1. one mention-scoped `non_character` record per own reference in `mention-scoped-audit.json` (a superseded decision
   stays in the record's `history`);
2. in `cast_preparation_progress.json`: the actor and its own exact aliases are removed from `registry`/`aliases`, an
   entry is **appended** to `source_reviewed_retirements` (evidence, own refs, reviewer, basis, the full retired
   registry entry, the removed aliases, registry digest before) and the input hash is appended to
   `source_retirement_inputs`;
3. append-only resolution of the actor's superseded blocked rows and of its `cast_quality` rows.

Original `characters.json`, `voices.json`, profiles and assets are never rewritten; history is never edited or removed.
An input file is processed **once** (its hash is recorded even when every entry is blocked), so a file left in place never
replays against a registry it no longer matches; to correct a blocked entry the caretaker writes a new file with the
current digest.
