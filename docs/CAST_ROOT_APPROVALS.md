# Caretaker root approval contract (`approve_root`)

A pending `variant_draft_root` (a line of `cast_pending_proposals.jsonl`, cited by an open
`context_variant_draft_root` row) becomes a registered cast actor **only** through this input. Nothing is
inferred or automatic: a root the file does not name is never touched.

Caretaker writes one complete `cast_root_approvals.json` snapshot by atomic rename, beside the project's other cast
files. The producer reads it at a cast batch boundary, right after the v2 context resolution file.

```json
{
  "contract": "cast_variant_root_approval",
  "version": 1,
  "source_sha256": "<full source book sha256>",
  "approvals": [{
    "action": "approve_root",
    "actor_id": "the root's actor_id",
    "proposal_sha256": "<exact sha256 of the stored pending proposal line>",
    "note": "caretaker rationale (must not claim a human review)",
    "reviews": [{
      "proposal_sha256": "<same sha>",
      "scope": {"chapter": "NN-part_NN.txt", "chapter_sha256": "...", "unit_id": "c00s00000",
                "quote_sha256": "...", "label": "exact label", "span_start": 0},
      "provenance": "native_review_audit | source_reviewed_trusted_role",
      "verdict": "distinct_living_identity",
      "reviewer_role": null,
      "factual_witnesses": [],
      "factual_basis": ""
    }]
  }]
}
```

`scope` is copied verbatim from the root's `variants[].scopes[].scope`; **one review per included mention, no more, no
fewer**. `proposal_sha256` is `proposal_sha(root)`: sha256 of the stored line (canonical `json.dumps(sort_keys=True,
ensure_ascii=False)`).

## Review provenance (truthful, exactly two kinds)

* `native_review_audit` — the mention already has a native schema-bound review in `new-identity-review-audit.json`
  whose verdict **and** `raw_review.verdict` are `distinct_living_identity`. `reviewer_role` must be `null`,
  `factual_witnesses` `[]`, `factual_basis` `""`: the audit record itself is the proof.
* `source_reviewed_trusted_role` — a trusted role (`caretaker`) read the exact source and states a factual basis.
  `reviewer_role` must be a trusted role, `factual_basis` non-empty, and `factual_witnesses` one or more exact
  immutable units (`chapter_file`, `chapter_sha256`, `unit_id`, `unit_quote`, `unit_quote_sha256`), at least one of which
  literally names the mention label or the identity. It is **never** labelled human-reviewed: any other provenance
  (e.g. `human_reviewed`) or any text claiming a human review is rejected. A source review cannot override a native
  verdict that says otherwise (e.g. `existing:<id>`, `uncertain`).

Native and source reviews may be mixed per mention.

## What the materializer guarantees

Fail closed (whole file, nothing written) on: wrong book sha, unknown contract/version, any extra/missing field, an
unknown `proposal_sha256`, an `actor_id` that is not the proposal's, a proposal that is not an *open pending* proposal
(superseded or resolved), any bio/look/source-fact/mention/witness/link that is not exact in the current source bytes
(re-proven, including the shared-unit or explicit kinship / continuous-participant links), or a review set that does not
cover the included mentions exactly once.

A root that fails a *gate* is blocked alone (typed `root_approval_blocked` pending row in stage `cast_root_approval`,
which blocks cast freeze; nothing else is written). All must pass:

1. `variant_root_approval_gate` — living-capable kind, no registry/alias duplicate, a bio citation that literally names
   the identity, a cited look (or the explicit no-visual-details sentinel), and an exact-scope
   `distinct_living_identity` review per mention (with its own-source witness).
2. Independent duplicate gate — never the narrator or an original anchor, no registered actor with the same name, no
   original `characters.json` / `voices.json` entry with the same id or name, no variant spelling that is already an alias
   of, or literally carried by, another registered actor.
3. Distinct-participant gate — a family-only label (mother, brother, auntie, ...) joins a root only while its bounded
   scene shows exactly one named participant (the root); two or more (e.g. two siblings) block the root. A lone
   sentence-initial capitalized word is not counted as a participant (lexically indistinguishable from a common word).
4. Every mention is an exact immutable name reference and has no conflicting scoped decision.

Only then, in this crash-safe order: (1) one mention-scoped `alias` record per included mention in
`mention-scoped-audit.json` (`approved_root` marks root, proposal sha and review provenance); (2) one new registry actor
(profile facts = the cited bio/look, `origin: approved_root`, the raw approval and proposal refs retained) plus the
exact `actor_id`/name aliases only — variant spellings are never global aliases — persisted in the progress file;
(3) append-only resolution of exactly this root's pending rows. Original `characters.json`, `voices.json`,
profiles and assets are never rewritten (freeze adds missing actors exactly as for any prepared actor).
Replaying the same file is a no-op: an already-registered exact proposal only re-asserts steps 1 and 3, and re-ingesting the
v2 file raises no new pending row for it.
