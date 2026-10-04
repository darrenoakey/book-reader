# Wide-bio mixed-profile recovery (`src/wide_bio_recovery.py`)

Offline recovery of the chunks a base wide-bio run could not finish, under a new provider profile
(`upstream-json-blanks32+presence-penalty-v1`, penalty 1.5, all other sampling/plan fields unchanged).

1. `seal` — read-only whole-line snapshot of the base run, offline revalidation by the existing `DeltaRun`
   under the ORIGINAL plan/profile/prefix state, hash-keyed `seal/manifest.json`
   (`old_validated_ready | old_truncated | missing`; anything else is `old_unresolved` and blocks).
2. `plan` — verifies the seal, lists targets (`old_truncated` + `missing`, never old-ready) and prints the caretaker command.
3. `recover` — resumable runner into `<workdir>/recovery/` (own journal; old records are never copied into it).
   Without `--execute` it only replays. `--execute` is the sole path that can send a request; it requires a valid
   calibration, a model listing advertising `capabilities.presence_penalty == presence-penalty-v1`, and every payload
   must equal the old-profile payload plus exactly `presence_penalty`. Never run by tests or agents other than the caretaker.
4. `verify` — union verifier: all chunks must be `verified_old_ready` or `verified_new_accepted`; then `state` is
   `source_coverage_ready` (only source coverage by verified chunks), otherwise `not_ready`. Pending rows may remain and candidate
   figures are counts/hashes only: `full_candidate_accounting` is always `not_verified` until the owner's existing gates run.
   It never freezes or applies a cast (`freeze_allowed` is always false); owner integration is never an automatic apply.

Exactly one attempt per target chunk: there is no `--max-attempts` (the attempt bound is not part of the plan fingerprint); retries need a fingerprinted design first.

Operational note: `seal`, `plan`, `verify` replay the full 246-chunk source (about 70 s each on the real run, tokenising ~9 MB);
run them detached with a log and sleep, never inside one long tool call.
