# Whole-book cast index (`src/cast_index.py`)

Deterministic, read-only support for whole-book cast preparation. It never calls a model, never writes a project,
and does not change `src/cast_freeze.py`. It reuses `immutable_evidence_units`, `immutable_name_references`,
`mention_scope`, `scoped_alias_proof`, `readjudication_due` and `adjudication_owners` unchanged, so ids, offsets and
scope hashes are identical to the existing ledger. It is **support for coverage and reconciliation**, not a substitute
for sending the whole book to the extractor.

Stages (each pure; the first three depend only on source bytes and registry/audit):

1. `build_source_index(chapters)` – every capitalized name occurrence with `(chapter_sha256, quote_sha256, label, span_start)`,
   per-chapter text/file hashes and char counts, a digest of everything. Fails closed (`CastDataIssue`) on an undecodable
   chapter or when units do not tile a chapter exactly.
2. `group_candidates(index)` – groups by the **exact label** (never by spelling: `Mary-Ann` and `Mary Ann` stay separate,
   where the ledger merges them). Same retention rule as the ledger, one pass; each group carries counts, `nonentity`
   and `scope_sha256`. Retained + dropped occurrences always equal all occurrences.
3. `resolve_known(...)` / `build_cast_index(...)` – partitions each group's occurrences: `literal_owner` (exact approved
   name/alias of a real cast member), `alias` (scoped record that `scoped_alias_proof` still proves), `non_character`,
   `ambiguous` (final), `stale` (must be re-decided: unproven alias, or an owner now plausible but never offered), `open`.
   A record applies only to its exact scope; changed bytes or offset never match.
4. `build_chunk_plan(index, max_chars)` – paragraph-boundary chunks covering every source character exactly once
   (verbatim segments with slice hashes; a paragraph is never split; oversize paragraph flagged). Independent
   `verify_chunk_coverage` recount; fails closed unless exactly-once. Every occurrence is placed in a chunk.
5. `reconcile(cast, claims, plan)` – `Claim(claim_id, names)` is plain data from any extractor. Reports per exact label:
   `covered` (a claim name equals it), `component` (whole-word part of a claimed name; a relation, not a merge),
   `settled` (already established), `omitted` (undecided and unclaimed; with factual diversified contexts and the chunks
   holding it); plus `unsupported_claim_names` (not a whole word anywhere in the source). Deterministic `digest`.

CLI (read-only, long on the real book – run detached): `.venv/bin/python -m src.cast_index output/<project> [--chunk-chars N]`.
