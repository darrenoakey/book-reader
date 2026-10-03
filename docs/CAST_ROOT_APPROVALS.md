# Caretaker root approval contract (`approve_root`)

A pending `variant_draft_root` (a line of `cast_pending_proposals.jsonl`, cited by an open
`context_variant_draft_root` row) becomes a registered cast actor **only** through this input. Nothing is
inferred or automatic: a root the file does not name is never touched.

Caretaker writes one complete `cast_root_approvals.json` snapshot by atomic rename, beside the project's other cast
files. The producer reads it at a cast batch boundary, right after the v2 context resolution file. `ingest_root_approvals` returns
`{"materialized": [], "already": [], "blocked": [], "malformed": []}`.

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
  (e.g. `human_reviewed`) or any text claiming a human review is rejected.

### Source review versus the native audit record of the same mention

A `source_reviewed_trusted_role` review is judged against the mention's record in `new-identity-review-audit.json`
(if any). The exact factual witnesses and every gate below must still prove it; the native record only decides whether
it may be resolved at all:

| native record for the exact scope | source review |
| --- | --- |
| none, or `distinct_living_identity` | accepted |
| `uncertain` (also a low-confidence or rejected/unsupported raw `existing:<id>` / `same_provisional:*` review) | **resolves it** |
| `existing:<id>` that is stale/unusable: `<id>` is not a registered actor (or is the narrator), or neither the support the record carries (`provenance`: owner name / profile anchor) is still honoured by the registry nor does the mention's current source scene re-prove the owner with the ordinary review guards | **resolves it** |
| `existing:<id>` that is **currently valid and source-supported** (same test; also an `uncertain` record whose raw review is such an `existing:<id>`) | **blocked** (`contradicts the native review verdict`) |
| any other verdict (`nonidentity_fragment`, `same_provisional:*`, unknown) | **blocked** (real conflict) |

A native-provenance review (`native_review_audit`) never rides on an `uncertain` record: it still needs a native
`distinct_living_identity` verdict **and** raw verdict. Resolving a native record never bypasses the source/hash
exactness checks, the independent duplicate / original-anchor gate, the distinct-participant gate, the exact-scope
scoped-decision conflict gate, or the exact immutable name-reference gate. When a source review resolves a non-distinct
native record, the mention-scoped alias record carries `approved_root.resolved_native_verdict` (the native verdict that
was resolved, e.g. `"uncertain"`) so the audit trail stays truthful.

Native and source reviews may be mixed per mention.

## What the materializer guarantees

Fail closed (`OperationalError`, whole file, nothing written) on file-level source/hash/contract problems: an
unreadable or non-object file, wrong book sha, unknown contract/version, any extra/missing top-level field or a
non-list `approvals`, a well-formed but unknown `proposal_sha256`, an `actor_id` that is not the proposal's, a proposal
that is not an *open pending* proposal (superseded or resolved), an unreadable pending-proposal store, or any
bio/look/source-fact/mention/witness/link (including a review's factual witness) that is not exact in the current source
bytes (re-proven, including the shared-unit or explicit kinship / continuous-participant links).

### Malformed individual entries (typed pending, never a crash)

An approval entry that is invalid **on its own** does not crash the producer. It is recorded as a pending row in stage
`cast_root_approval`, code `root_approval_malformed`, item `root_approval_malformed:<entry_sha256[:16]>`, evidence
`{"entry_sha256": payload_hash(entry), "actor_id": <str or null>, "problem": "<why>"}`. Malformed means: the entry is not
an object with exactly `action, actor_id, proposal_sha256, reviews, note`; an `action` other than `approve_root`;
non-text `actor_id`/`note` or a `proposal_sha256` that is not 64 lower-case hex characters; text claiming a human review;
a repeated `actor_id` / `proposal_sha256` in the file; or `reviews` that are not a non-empty list of exactly the documented
review schema covering every included mention exactly once (review citing another proposal sha, non-text or non-matching
scope, provenance/verdict other than the two documented, a native review carrying a role/witnesses/basis, a source
review without a trusted role, without a non-blank basis, or without witnesses of the documented schema).

While any entry is malformed the file is **held whole**: no root of it is materialized or gate-blocked, nothing else is
written, and the rows (which block cast freeze like every pending row of the stage) say so. When the file changes so the
entry is gone (or the file is removed) the row is resolved `root_approval_entry_superseded`; replays do not stack rows.

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
