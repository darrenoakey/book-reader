# greenline — book-reader

This repo uses **greenline**: a local, serialized, gated CI/CD tool. There is one
canonical checkout of `book-reader`; it is always on `main`, always clean,
and always identical to what prod (`book-reader`) runs. Every change reaches
`main` through a single quality gate that runs change-impact validation and a
real production deploy, one submission at a time.

## The workflow

```
greenline worktree feature-x        # new worktree at /Volumes/Gumby/worktrees/greenline/book-reader/feature-x
cd /Volumes/Gumby/worktrees/greenline/book-reader/feature-x         # branch gl/feature-x off last-green
# ...edit, test, commit...
greenline submit                     # gate: squash-merge -> check -> ff main -> deploy -> publish
greenline done                       # remove the worktree + branch once merged
```

Diagnostics (no lock needed):

```
greenline status     # lock holder, SHA drift (main/origin/last-green/deployed), recent journal
greenline doctor     # invariant checks; `greenline doctor --fix` reconciles under the lock
```

If commits reached `main` **outside the gate** (a legacy-workflow session,
a hotfix), greenline refuses to run until they are gated — it never discards them.
Gate them in place with:

```
greenline adopt      # check + deploy the current main tip; never resets main
```

## What the gate does (`greenline submit`)

Under an exclusive lock (other submits queue behind it):

1. **Preflight reconcile** — recover any crashed prior gate from the journal;
   confirm the canonical checkout is clean and on `main`; reconcile
   `main`/last-green/origin; reset the gate worktree to the main tip.
2. **Squash-merge** your branch in the gate worktree. Conflict or empty diff → fail fast.
3. **Check** — run `./run check` with cwd = the gate worktree. It validates every
   behavior the change could break, selected by changed paths and transitive impact.
4. **Advance** — fast-forward the canonical `main` to the candidate.
5. **Deploy** — run `./run deploy` with cwd = the canonical checkout. Nonzero → **automatic rollback**: main resets to the previous commit and `./run deploy` re-runs to restore prod.
6. **Publish** — push `main` to origin (via a one-shot allow-push flag the pre-push hook honours) and advance `last-green`.

## Release small, keep going

Choose the smallest coherent, useful change, validate its relevant impact, and
release it immediately; then take the next chunk. Prioritize a working, deployed
fix for any reported production blocker. Do not broaden ready-to-ship scope or
bundle speculative improvements. A release boundary is never a permission pause:
continue the whole authorized goal without asking. Impact selection must become
smarter as the product grows so growth does not increase release latency.

## The contract (you must implement in `./run`)

- **`./run check`** — cwd = the worktree being gated. Start with all useful, relevant
  verification that fits in three minutes, then use changed paths and transitive
  impact to prove the selection covers every behavior the candidate could affect
  against a **test** datastore. Preserve relevant coverage; run broader or external
  suites only when intentionally relevant. Product growth must not increase standard
  release time. Exit code is the verdict. Must run concurrently from multiple worktrees.
- **`./run deploy`** — cwd = the canonical checkout. Rebuild/restart prod (e.g.
  `auto -q restart book-reader`), **health-check**, exit nonzero on unhealthy, and be
  **idempotent** (rollback re-runs it).
- **`./run health`** *(optional)* — probe only. If absent, greenline re-runs `./run deploy` as the health probe.

The full release normally averages and targets 180 seconds and has one monotonic
600-second loaded-machine worst-case hard deadline covering admission, reconcile,
check, deploy, publish, and attestation. At five minutes, investigate the active
stage immediately. At ten minutes, terminate and reap the attempt, then fix its
cause before submitting again—never wait for load, rearm a TTL watcher, or retry
unchanged. Avoid narrower inner timeouts that falsely fail useful work while
aggregate time remains.
Release stages may use the actual time remaining; there are no narrower arbitrary
caps. Recovery begins only after failure under a separate bound and is never called
release success. Every successful submission deploys; legacy `coalesce_deploys`
config is ignored. Queue/check/deploy/publish/total timings are journaled and
printed; a green total over 180 seconds automatically launches a detached speed-up
investigation after unlock, and acceptance is reported only after its tool proves it.

## Test/code co-design (from docs/DOCTRINE.md — read it)

Tests run in parallel with each other, with other agents' runs, and with live prod.
Therefore: namespace every entity a test creates (uuid suffixes); never assert global
state (counts, "table empty", singletons); never truncate/reset shared stores; tolerate
pre-existing and concurrently-changing data; bind servers to OS-assigned ports (port 0);
probe the exact host:port you own (TCP dial / unix connect) instead of `lsof`, `netstat`, `ss`, `fuser`, or other system-wide scans;
prefer per-test datastores (tmp sqlite, worktree-relative `local/`); keep schema
migrations backward-compatible one version (deploy rollback runs the previous code against
the new schema); serialize tests that share a scarce resource (a local LLM server, one GPU).

**Real but fast:** never mock another service — but make real tests quick with a
content-addressed record/replay cache: route all access to the external service through
one choke point; key = hash(endpoint + model + full request); store `{model, request,
response}` JSON per key in a gitignored `local/<service>-cache/` (idempotently created,
atomic writes, corrupt entry = self-healing miss). Miss → real call; hit → the real
recorded response in milliseconds. First run is genuinely end-to-end; every rerun is
instant, and the persistent gate worktree keeps the gate's cache warm. To re-record,
delete the cache dir. Don't cache calls whose variability is what's under test.

## State (in the shared git dir)

`.git/greenline/`: `lock`, `journal.jsonl`, `status.json`, `deployed` (SHA prod runs),
`logs/`, `allow-push` (transient, gate publish), `allow-main` (transient, PID-bound,
gate main-ref mutations). Ref `refs/greenline/last-green` = last fully-gated,
deployed-healthy commit.

**Main is hard-locked.** `greenline setup` installs three hooks:
- `reference-transaction` — refuses any local update of `refs/heads/main`
  unless `allow-main` exists and names a still-alive PID. Cannot be bypassed with
  `--no-verify`. This is what actually prevents working on main.
- `pre-commit` — early reject when HEAD is on `main` (friendly message;
  bypassable — the reference-transaction hook is the real lock).
- `pre-push` — refuses direct pushes to `main` unless the gate sets
  `allow-push`.
