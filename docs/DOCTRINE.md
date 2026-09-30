# Greenline Doctrine

This is the master copy. `greenline setup` installs it into each repo. It is the
philosophical core of the gate: **main is always green and always deployed, one
serialized gate makes it so, and tests and code are co-designed so the gate can
run for real, in parallel, against live prod.**

## The five invariants

1. **`main` == deployed == green, always.** Every commit on `main` has passed the
   release validation selected for that change and been deployed healthy. There is no "deployed but
   untested" state. `last-green` records the last commit that passed the whole
   gate; `deployed` records the SHA prod runs.

   There is no queued-load exception: a successful submission attests its own
   deploy before returning. The legacy `coalesce_deploys` config key is accepted
   for compatibility but ignored.

2. **The canonical checkout is pristine.** It lives at the repo root (the parent of
   the shared git dir). It is always on `main`, always clean. **No human and no
   agent ever edits it, ever.** Only greenline writes to it — to fast-forward `main`
   during a successful gate, or to reset+redeploy during a rollback. If the
   canonical checkout is dirty, the gate refuses to run and tells the operator to
   resolve it — greenline never silently discards uncommitted work.

3. **All work happens in worktrees branched from last-green.** `greenline worktree
   NAME` creates `gl/NAME` off `last-green` (the known-good tip), in its own
   worktree. Agents work there in isolation. Many worktrees coexist; many agents
   work at once.

4. **Every merge goes through the serialized gate.** One exclusive lock. A submit
   squash-merges the candidate, runs the **real** `check`, fast-forwards `main`,
   runs the **real** `deploy`, and publishes — atomically from the operator's point
   of view. If check fails, `main` is untouched. If deploy fails, prod is rolled
   back to the previous commit and redeployed. No partial states leak out.

5. **Test data never touches prod datastores.** `check` runs against a test
   datastore; `deploy` touches prod. The two never cross. A test that writes to the
   prod store is a bug in the test, not a flaky gate.

## Release small, keep going

Choose the smallest coherent, useful change, validate its relevant impact, and
release it immediately; then take the next chunk. Prioritize a working, deployed
fix for any reported production blocker. Do not broaden ready-to-ship scope or
bundle speculative improvements. A release boundary is never a permission pause:
continue the whole authorized goal without asking. Impact selection must become
smarter as the product grows so growth does not increase release latency.

## Test/code co-design

The gate runs `check` for real, and `deploy` restarts a **live** service that is
serving other work. So tests are not written in a vacuum — the code they test and
the tests themselves are designed **together** to survive this environment:

- **Tests run in parallel** — with each other, with other agents' `check` runs from
  other worktrees, and with a live prod instance. Nothing may assume it is alone.
- **Namespace every entity a test creates.** Unique IDs/prefixes per run (a uuid
  suffix on names, keys, table rows, temp dirs). Two `check` runs at once must never
  collide.
- **Never assert global state.** No "row count == N", no "the table is empty", no
  "this singleton does/doesn't exist". Other runs and prod are changing the world
  underneath you. Assert only about the entities *this run* created.
- **Never truncate or reset shared tables/stores.** No `TRUNCATE`, no `DROP`, no
  "clean slate" fixtures on anything shared. Create your own namespaced data and
  clean up only that.
- **Tolerate pre-existing and concurrently-changing data.** Reads may return other
  runs' rows; filter to your namespace. Counts drift between two reads; don't depend
  on them.
- **Bind servers to OS-assigned ports (port 0), never fixed ports.** Two `check`
  runs binding `:8080` is an instant, confusing failure. Ask the OS for a port and
  read back what you got.
- **Probe the exact host:port (or unix socket) you own. Never scan the machine.**
  `lsof`, `netstat -a`/`-anv`, `ss`, `fuser`, `lsof +D`, and `ps -ax`/`ps aux` are
  system-wide scans: they burn CPU, race with parallel checks, and can mis-attribute
  unrelated processes. Dial or connect the specific address, or bind port 0 and read
  the assigned port back. If you need a process identity, read that one PID (pidfile
  you wrote, `/proc/<pid>`, `kern.proc.pid`). If you cannot identify a listener
  without a machine-wide scan, fail closed and leave it alone. Do not replace one
  scan with a worse one.
- **Prefer per-test / per-worktree datastores where cheap.** A `tmp_path` sqlite
  file, a worktree-relative `local/` dir — isolation by construction beats
  careful namespacing.
- **Serialize on scarce shared resources.** A single local LLM inference server or
  one GPU is a serialization point: design those tests to tolerate latency and to
  run one-at-a-time (`pytest -p 1` for that group, or an explicit lock), and mark
  and document them so the constraint is visible.
- **Schema migrations must be backward-compatible one version.** Old code must run
  against the new schema, because a deploy rollback **redeploys the previous code
  against the already-migrated store** — it does not un-migrate. So never rename or
  drop a column the previous release still reads in the same deploy that starts
  writing the new one; do it in two releases.
- **`check` must be runnable concurrently from multiple worktrees.** This is the
  summary constraint that all the above serve. If two agents cannot both run
  `./run check` at the same moment without interfering, the check is broken.

## The release time budget

**A normal complete release averages and targets 180 seconds; 600 seconds is the
loaded-machine worst-case hard failure deadline, not the target.** That one
monotonic deadline runs from command invocation through publish. Lock admission, crash reconcile,
candidate construction, change-impact validation, deploy, publish, and
completion attestation all consume that same clock. The 180-second target is a
performance signal only: crossing it never rejects a release that completes
correctly before 600 seconds. The ordinary flock stays deliberately simple;
there is no FIFO scheduler, load gate, coalesced success, or deferred deploy.

Release stages do not carry narrower arbitrary caps: each may use the actual
time remaining in the aggregate 600-second budget. A timeout kills and reaps the
whole subprocess group, records a terminal release failure, and never labels the
expired release successful. Recovery then starts under its own separately
bounded clock, so an expired success deadline cannot cut off restoration of the
previous deployed version. Recovery success is reported only as recovery, never
as release success. A standalone `health` command remains capped at five seconds
because a readiness probe must be a probe, not a build.

Every successful release journals and prints queue, check, deploy, publish, and
total timings. A total over 180 seconds stays green and automatically starts a
detached `greenline-speedup` investigation after the lock is released. Greenline
reports whether that investigation was accepted, failed to launch, or is still
pending; merely creating a launcher process is never reported as acceptance.

At five minutes, immediately inspect the active stage and its process tree: the
release is already far outside its normal envelope. Do not wait for machine load,
start a TTL watcher, or rearm the attempt. At 600 seconds greenline terminates and
reaps that attempt. Diagnose and fix the cause before a new submission; never
retry an unchanged candidate merely because the machine may be quieter.

Validation design starts with **all useful, relevant verification we can fit in
three minutes**. Dependency and ownership maps, changed-file selection, and
transitive impact analysis must then prove that the selected validation is
complete for behavior the candidate could affect. Preserve existing coverage
where it remains relevant; run broader or external suites only when the change
makes them intentionally relevant. Product growth must not increase standard
release time. Do not create false failures with inner stage or operation timeouts
shorter than the remaining release budget. The remaining remedies are:

- **Parallelize.** The co-design rules above exist so tests can run at once —
  isolated schemas, namespaces, ports and temp dirs per test mean nothing has to
  be serialized for correctness. Run them wide.
- **Start from a known position.** Prebuilt fixtures, seeded snapshots, a
  restore-from-template database — never rebuild the world per test.
- **Keep the build warm.** The gate worktree is persistent on purpose: leave
  build/dependency caches (`target/`, `.venv`, node_modules, record/replay
  caches) in place so a submission compiles only its own diff.
- **Validate by impact.** Select the affected build, lint, unit, integration,
  migration, and UI checks from the changed paths and their transitive consumers.
  Cheap and decisive checks run first; unrelated product areas do not run.
- **Split monolithic integration tests.** One 4-minute end-to-end test is a
  single-threaded wall; several focused ones run concurrently.
- **Cache the external world** — see the next section.

## Real but fast: record/replay caches for external services

Tests must be **real** — mocking, faking, or stubbing another service is forbidden,
full stop. But real does not have to mean slow. The dominant cost in a real test
suite is usually calls to external services (an LLM inference server, a third-party
API). You cannot fake those services — but you **can assume an identical request
will get the same answer it got last time**, and cache accordingly:

- **Route all access to the external service through one choke point** in the app
  (one client function/module). The layer looks identical to callers.
- **Inside it, a content-addressed cache**: key = hash of a canonical serialization
  of (endpoint kind + model/service + the FULL request — everything semantically
  meaningful, including options like temperature). Store one human-readable JSON
  file per key — `{model, request, response}` — in a gitignored local cache dir
  (e.g. `local/<service>-cache/`), created idempotently on first use.
- **Miss → real call**, then write the entry atomically (temp file + rename).
  **Hit → the stored response, zero network.** A corrupt/partial entry is a miss
  that self-heals on rewrite.
- This is **not a mock**: every cached byte came from the real service. The first
  run of any test is a genuine end-to-end call; every rerun replays the real
  recorded answer in milliseconds.

Consequences: prod is unaffected (real traffic is effectively always novel, and a
cache-layer failure must never break the live call); test suites collapse from
minutes to seconds after their first run; and because the **gate worktree is
persistent**, the gate's cache stays warm across submissions — the serialized merge
gate stays fast. To re-record reality (service upgraded, model changed), delete the
cache dir; the next run repopulates it from live calls.

Caveat: don't cache a call whose *variability* is the thing under test (e.g.
sampling diversity), and always include every request field that changes the answer
in the key. Reference implementation: darrennn's `src/darrennn/endpoint_health.py`.

## The contract

- **`./run check`** — cwd = the worktree being gated. Builds, lints, and runs all
  validation selected by the candidate's real change impact against a TEST
  datastore. Its **exit code is the verdict** (0 = green). Must be safe to run
  concurrently from multiple worktrees.

- **`./run deploy`** — cwd = the canonical checkout. Rebuilds and restarts prod
  (e.g. `auto -q restart <svc>`). It **MUST health-check and exit nonzero on
  unhealthy**, and it **MUST be idempotent** — greenline re-runs it during rollback
  and as the default health probe, so running it twice must be safe.

- **`./run health`** *(optional)* — a probe only, no side effects. If absent,
  greenline re-runs `deploy` as the health probe (which is why `deploy` must be
  idempotent).
