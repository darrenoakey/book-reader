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

## Casting, annotated imports, and verification

- `./run noveliser PROJECT --import-only` accepts the noveliser's finished folder, preserving its supplied speakers and text without LLM re-attribution. `./run noveliser PROJECT` continues through narration/movie production. Script-only minor characters are real declared identities even when absent from the supplementary character sheet; never map them to narrator.
- `cast_plan.json` is the shared evidence-based casting plan. Accent follows formative background, not the current scene/address: an American visiting France stays American; French locals retain their accents. Shared-upbringing cohorts preserve supported individual exceptions. Never infer acoustic traits from names or race.
- `voices_plan.json` / `kokoro_voices_plan.json` bind descriptions/mappings to both plan contents and actual artifact bytes. Existing materialized clips/audio fail closed on incompatible recasting; archive/reimport explicitly rather than mix old and new voices. Kokoro cannot provide French-accented English by substituting its French-language pack; use the default voice-design engine for unsupported accents.
- Worktrees and the gate need the machine's authoritative gitignored `local/config.toml`, not silent default models. On this machine canonical text routing is Tensorfold `qwen3.6:35b-a3b` on Boringstack `:11436/v1`; scriptor uses native Ollama `scriptor:v5` on `:11434`. Preserve configured identities when repairing services.
- **First-time real casting recording suites, full integration qualification, model downloads and backend initialization must run detached with durable logs/verdict files; use agent sleep between bounded observations.** Never hold an agent turn while they run, poll in a loop, or repeatedly retry an unchanged failed gate. Warm content-addressed casting recordings make subsequent gates fast.
- Historical frozen-cast QA must use the original byte-identical source prefix from the append ledger when the live source was extended. Verify its original SHA; never update the historical assertion to bless a different text.
- Ruff is pinned in `requirements.txt`: minor-version differences changed `tomllib` import classification between worktree and gate. Qualify with the pinned version; never suppress lint or mix tool versions to claim green.
