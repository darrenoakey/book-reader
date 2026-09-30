<!-- >>> greenline >>> -->
## Greenline gate — how merges work here

This repo is gated by **greenline**. Read `docs/greenline.md` and `docs/DOCTRINE.md`
before writing code or tests.

**Invariants (never violate):**
- `main` == what prod runs == green, always.
- The canonical checkout is pristine — never edit it by hand.
- All work happens in worktrees branched from last-green.
- Every merge goes through the serialized gate: change-impact `check` + real `deploy`.

**Your workflow:**
1. `greenline worktree <name>` — get a worktree at `/Volumes/Gumby/worktrees/greenline/book-reader/<name>` on branch `gl/<name>`.
2. Do your work there. Co-design tests + code per docs/DOCTRINE.md (parallel-safe, namespaced, no global-state assertions, OS-assigned ports, probe the exact host:port instead of machine-wide `lsof`/`netstat`/`ss`/`fuser` scans; never mock other services — make real calls fast with a content-addressed record/replay cache).
3. Commit in your worktree. Then `greenline submit` (from that worktree).
4. The gate squash-merges, runs `./run check`, fast-forwards `main`, runs `./run deploy`, and publishes. It rolls back prod automatically if deploy fails.
5. On success: `greenline done` to remove your worktree + branch.

**Never** commit or push on `main` — hooks hard-lock it (reference-transaction
cannot be bypassed with `--no-verify`; pre-commit/pre-push refuse too). Never edit the
canonical checkout. If the gate reports a conflict, rebase your worktree on
`main` and resubmit. If commits somehow reached `main` outside the
gate (legacy workflow, hotfix), run `greenline adopt` to gate them in place —
greenline never discards commits on `main`.

Diagnose with `greenline status` and `greenline doctor` (`--fix` to reconcile).
<!-- <<< greenline <<< -->
