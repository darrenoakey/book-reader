"""Fingerprint-safe mixed-profile source recovery for a wide-bio delta run.

A finished-or-growing base run (one provider profile) holds three kinds of chunk: validated-ready, length-truncated and never-asked.  This module recovers the last two under a NEW provider profile (the presence-penalty profile) without re-asking or rewriting anything the base already proved:

  seal    read-only consistent snapshot of the base run + offline revalidation of every saved response under the ORIGINAL plan, profile and prefix state (existing DeltaRun), written as a hash-keyed manifest in a new work directory.
  plan    offline: verify the seal and list exactly the chunks the recovery would ask (old_truncated + missing, never old_validated_ready).
  recover resumable runner into its own journal; ONLY `--execute` can send a request, it first proves the server advertises the presence-penalty capability and that every request differs from the old-profile request only by the penalty.  Offline it just replays.
  verify  offline union verifier: every chunk of the source must carry one verified successful disposition (old revalidated claims or newly accepted evidence) with source/candidate accounting; anything else is not_ready.  Nothing here ever freezes or applies a cast.

Old records are never moved into the new-profile journal and the base directory is never written.  No cap, prompt or validator is changed: the old plan fingerprint is rebuilt byte-for-byte and the new plan differs only by the profile descriptor and the presence penalty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.cast_index import build_cast_index
from src.llm import request_for
from src.wide_bio import (
    HARD_DEADLINE_S,
    SOFT_DEADLINE_S,
    ContractError,
    Counter,
    ProofConfig,
    Transport,
    build_counter,
    canonical_json,
    chat_transport,
    check_output_dir,
    load_proof_config,
    project_chapters,
    read_source,
    sha256_text,
    validate_calibration,
    write_atomic,
)
from src.wide_bio_delta import (
    NO_RESPONSE,
    DeltaChunk,
    DeltaPlan,
    DeltaRun,
    DeltaSettings,
    Prepared,
    build_delta_plan,
    run_lock,
)
from src.wide_bio_tokenizer import TokenizerRefusal

RECOVERY_VERSION = 1
OLD_PROFILE = "upstream-json-blanks32"
NEW_PROFILE = "upstream-json-blanks32+presence-penalty-v1"
PRESENCE_PENALTY = 1.5
CAPABILITY = "presence-penalty-v1"

OLD_READY = "old_validated_ready"
OLD_TRUNCATED = "old_truncated"
MISSING = "missing"
OLD_UNRESOLVED = "old_unresolved"
TARGET_DISPOSITIONS = frozenset({OLD_TRUNCATED, MISSING})

SNAPSHOT_TRIES = 5
BASE_FILES = ("plan.json", "seed.json")
DERIVED_FILES = ("claims.jsonl", "pending.jsonl")


# ##################################################################
# hashing helpers
def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(value) -> str:
    return sha256_text(canonical_json(value))


def refuse(message: str, code: str):
    raise ContractError(message, code)


def raw_names(chunk_id: str, attempt: int) -> tuple[str, str]:
    stem = f"{chunk_id}.a{attempt}"
    return f"{stem}.response.txt", f"{stem}.meta.json"


# ##################################################################
# context
# everything the offline checks need, rebuilt from the explicit source, project chapters, proof config and the base run's own plan.json/seed.json.  The old plan must reproduce byte-for-byte; a base that does not use the old profile, or already carries a presence penalty, is refused.
@dataclass(frozen=True, slots=True)
class Context:
    config: ProofConfig
    chapters: tuple[Path, ...]
    source: str
    source_sha256: str
    count: Counter
    stored_plan: dict
    seed: dict
    old_plan: DeltaPlan
    new_plan: DeltaPlan


def settings_from_stored(stored: dict, profile: str | None, penalty: float | None) -> DeltaSettings:
    sampling = stored.get("sampling") or {}
    return DeltaSettings(
        stored["target_tokens"],
        stored["delta_tokens"],
        1,
        sampling.get("temperature"),
        sampling.get("seed"),
        profile,
        stored.get("wire_format") == "compact-v2",
        penalty,
    )


def load_context(
    project: Path,
    source_path: Path,
    config_path: Path,
    plan_json: bytes,
    seed_json: bytes,
    expect_source_sha256: str,
    expect_base_plan_sha256: str,
) -> Context:
    try:
        stored, seed = json.loads(plan_json), json.loads(seed_json)
    except ValueError as error:
        refuse(f"base plan/seed are not JSON: {error}", "base_invalid")
    if not isinstance(stored, dict) or not isinstance(seed, dict):
        refuse("base plan/seed must be JSON objects", "base_invalid")
    if (
        stored.get("provider_grammar_profile") != OLD_PROFILE
        or "presence_penalty" in stored
        or not isinstance(stored.get("sampling"), dict)
    ):
        refuse("the base plan is not the sampled old-profile plan without a penalty", "base_profile_mismatch")
    if stored.get("plan_sha256") != expect_base_plan_sha256:
        refuse("base plan fingerprint is not the expected one", "base_plan_mismatch")
    config = load_proof_config(config_path)
    source_bytes = source_path.read_bytes()
    if sha_bytes(source_bytes) != expect_source_sha256 or stored.get("coverage", {}).get("source_sha256") != expect_source_sha256:
        refuse("source hash is not the expected one", "source_mismatch")
    source = read_source(source_path)
    chapters = tuple(project_chapters(project))
    count = build_counter(config)
    overhead = stored["fixed_overhead_tokens"]
    seed_sha = sha_json(seed)
    old = build_delta_plan(source, chapters, config, count, settings_from_stored(stored, OLD_PROFILE, None), overhead, seed_sha)
    if old.artifact != stored:
        refuse("the rebuilt old plan is not identical to the base plan.json", "base_plan_mismatch")
    new = build_delta_plan(source, chapters, config, count, settings_from_stored(stored, NEW_PROFILE, PRESENCE_PENALTY), overhead, seed_sha)
    differing = {key for key in old.artifact if old.artifact[key] != new.artifact.get(key)} | (set(new.artifact) - set(old.artifact))
    if differing != {"provider_grammar_profile", "presence_penalty", "plan_sha256"}:
        refuse(f"new plan differs from the old plan in more than the profile and penalty: {sorted(differing)}", "new_plan_drift")
    return Context(config, chapters, source, expect_source_sha256, count, stored, seed, old, new)


# ##################################################################
# consistent snapshot
# the base journal is append-only and hash-chained, so its longest whole-line prefix is a consistent cut: every record's raw response and meta are checked against the hashes the record carries, and the journal is re-read to prove it only grew.  Nothing is written to the base.
def parse_chain(prefix: bytes) -> list[dict]:
    records, previous = [], "0" * 64
    for number, line in enumerate(prefix.decode("utf-8").splitlines()):
        try:
            record = json.loads(line)
        except ValueError:
            refuse(f"base journal line {number} is not JSON", "base_journal_corrupt")
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if record.get("seq") != number or record.get("prev") != previous or record.get("record_sha256") != sha_json(body):
            refuse(f"base journal record {number} breaks the hash chain", "base_journal_corrupt")
        previous = record["record_sha256"]
        records.append(record)
    return records


def read_consistent_base(base: Path) -> dict[str, bytes]:
    """name -> bytes of plan.json, seed.json, journal.jsonl (whole lines only), every raw file the journal names, and the derived claims/pending files (reference only)."""
    for _ in range(SNAPSHOT_TRIES):
        try:
            journal = (base / "journal.jsonl").read_bytes()
            files = {name: (base / name).read_bytes() for name in BASE_FILES}
            prefix = journal[: journal.rfind(b"\n") + 1]
            records = parse_chain(prefix)
            for record in records:
                if record["status"] in NO_RESPONSE:
                    continue
                response_name, meta_name = raw_names(record["chunk_id"], record["attempt"])
                response = (base / "raw" / response_name).read_bytes()
                meta = (base / "raw" / meta_name).read_bytes()
                if sha256_text(response.decode("utf-8")) != record["response_sha256"] or sha256_text(meta.decode("utf-8")) != record["meta_sha256"]:
                    refuse(f"saved response of {record['chunk_id']} does not match its journal hashes", "base_raw_mismatch")
                files[f"raw/{response_name}"], files[f"raw/{meta_name}"] = response, meta
            derived = {}
            for name in DERIVED_FILES:
                path = base / name
                derived[name] = path.read_bytes() if path.is_file() else b""
            grown = (base / "journal.jsonl").read_bytes()
        except OSError as error:
            refuse(f"base run directory is unreadable: {error}", "base_unreadable")
        if grown.startswith(prefix):
            return {**files, "journal.jsonl": prefix} | {f"derived/{k}": v for k, v in derived.items()}
    refuse("the base journal did not stay append-only across snapshot attempts", "base_journal_unstable")


def write_snapshot(files: dict[str, bytes], dest: Path) -> None:
    """Idempotent: a present file must be byte-identical, an absent one is created."""
    for name, data in sorted(files.items()):
        path = dest / name
        if path.is_file():
            if path.read_bytes() != data:
                refuse(f"sealed snapshot file {name} differs from the base snapshot", "seal_mismatch")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


# ##################################################################
# offline revalidation of the old run
# the existing DeltaRun replays the sealed copy under the ORIGINAL plan, profile and prefix state: every saved raw response is re-judged against rebuilt state and must reproduce its journal record.  Only the sealed copy is ever replayed, so the base is never opened for writing.
def replay_old(ctx: Context, snapshot: Path) -> DeltaRun:
    cast = build_cast_index(list(ctx.chapters), ctx.seed.get("registry", {}), ctx.seed.get("aliases", {}), None)
    run = DeltaRun(list(ctx.chapters), ctx.config, ctx.old_plan, snapshot, ctx.count, cast, ctx.seed.get("registry"), ctx.seed.get("aliases"))
    run.replay()
    return run


def disposition(run: DeltaRun, chunk: DeltaChunk) -> str:
    if chunk.id in run.ready:
        return OLD_READY
    attempts = run.attempts[chunk.id]
    if not attempts:
        return MISSING
    return OLD_TRUNCATED if run.length_proven(chunk) else OLD_UNRESOLVED


def attempt_evidence(record: dict) -> dict:
    keys = ("attempt", "status", "called", "request_sha256", "response_sha256", "meta_sha256", "result_sha256", "record_sha256", "server")
    return {key: record.get(key) for key in keys}


def span_sha256(ctx: Context, chunk: DeltaChunk) -> str:
    return sha256_text(ctx.source[chunk.start : chunk.end])


def validation_evidence(validation: dict) -> dict:
    claims, pending = validation["claims"], validation["pending"]
    return {
        "claims": len(claims),
        "pending": len(pending),
        "duplicates": len(validation["duplicates"]),
        "claims_sha256": sha_json(claims),
        "pending_sha256": sha_json(pending),
        "claim_ids_sha256": sha_json([claim["claim_id"] for claim in claims]),
        "witness_sha256": sha_json([claim["witness"] for claim in claims]),
    }


def chunk_entry(ctx: Context, run: DeltaRun, chunk: DeltaChunk) -> dict:
    entry = {
        "id": chunk.id,
        "start": chunk.start,
        "end": chunk.end,
        "span_sha256": span_sha256(ctx, chunk),
        "paragraph_span": [chunk.paragraphs[0].id, chunk.paragraphs[-1].id],
        "base_user_sha256": next(item["base_user_sha256"] for item in ctx.old_plan.artifact["chunks"] if item["id"] == chunk.id),
        "disposition": disposition(run, chunk),
        "old_attempts": [attempt_evidence(record) for record in run.attempts[chunk.id]],
    }
    if chunk.id in run.ready:
        entry["old_validation"] = validation_evidence(run.ready[chunk.id])
    return entry


def derived_agrees(run: DeltaRun, files: dict[str, bytes]) -> dict:
    """Informational only: do the base's own claims/pending files equal what the replay derives (they are rewritten after the journal, so a growing base may lag)."""
    ordered = [run.ready[chunk.id] for chunk in run.plan.chunks if chunk.id in run.ready]
    claims = "".join(canonical_json(claim) + "\n" for item in ordered for claim in item["claims"])
    pending = "".join(canonical_json(row) + "\n" for item in ordered for row in item["pending"])
    return {
        "claims_jsonl": files.get("derived/claims.jsonl", b"") == claims.encode(),
        "pending_jsonl": files.get("derived/pending.jsonl", b"") == pending.encode(),
    }


def profile_block(plan: DeltaPlan) -> dict:
    return {
        "profile": plan.artifact.get("provider_grammar_profile"),
        "presence_penalty": plan.artifact.get("presence_penalty"),
        "temperature": plan.settings.temperature,
        "seed": plan.settings.seed,
        "delta_tokens": plan.settings.delta_tokens,
        "target_tokens": plan.settings.target_tokens,
        "output_tokens": plan.artifact["output_tokens"],
        "num_ctx": plan.artifact["num_ctx"],
        "plan_sha256": plan.artifact["plan_sha256"],
    }


# ##################################################################
# seal
def build_manifest(ctx: Context, run: DeltaRun, files: dict[str, bytes]) -> dict:
    chunks = [chunk_entry(ctx, run, chunk) for chunk in ctx.old_plan.chunks]
    counts = {name: sum(1 for item in chunks if item["disposition"] == name) for name in (OLD_READY, OLD_TRUNCATED, MISSING, OLD_UNRESOLVED)}
    journal = files["journal.jsonl"]
    records = run.journal.records
    body = {
        "recovery_version": RECOVERY_VERSION,
        "source_sha256": ctx.source_sha256,
        "source_chars": len(ctx.source),
        "base_plan_sha256": ctx.old_plan.artifact["plan_sha256"],
        "old_profile": profile_block(ctx.old_plan),
        "new_profile": profile_block(ctx.new_plan),
        "journal": {"records": len(records), "head": records[-1]["record_sha256"] if records else None, "sha256": sha_bytes(journal)},
        "files": {name: sha_bytes(data) for name, data in sorted(files.items())},
        "derived_agrees": derived_agrees(run, files),
        "dispositions": counts,
        "targets": [item["id"] for item in chunks if item["disposition"] in TARGET_DISPOSITIONS],
        "chunks": chunks,
    }
    return body | {"manifest_sha256": sha_json(body)}


def seal(ctx: Context, base: Path, workdir: Path) -> dict:
    base_resolved, work_resolved = base.resolve(), workdir.resolve()
    if work_resolved == base_resolved or base_resolved in work_resolved.parents or work_resolved in base_resolved.parents:
        refuse("the work directory must be a new directory outside the base run", "workdir_invalid")
    check_output_dir(workdir, list(ctx.chapters))
    files = read_consistent_base(base)
    if files["plan.json"] != (base / "plan.json").read_bytes() or sha_json(json.loads(files["seed.json"])) != ctx.old_plan.artifact["seed_sha256"]:
        refuse("base plan/seed changed while sealing", "base_plan_mismatch")
    snapshot = workdir / "seal" / "snapshot"
    manifest_path = workdir / "seal" / "manifest.json"
    if manifest_path.is_file():
        refuse("this work directory is already sealed; use a new one", "seal_exists")
    write_snapshot(files, snapshot)
    run = replay_old(ctx, snapshot)
    manifest = build_manifest(ctx, run, files)
    write_atomic(manifest_path, json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def verify_seal(ctx: Context, workdir: Path) -> tuple[dict, DeltaRun]:
    """Re-prove a seal from its own files: manifest hash, snapshot file hashes, and a fresh offline replay that must reproduce every chunk entry."""
    try:
        manifest = json.loads((workdir / "seal" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        refuse(f"seal manifest is unreadable: {error}", "seal_missing")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != sha_json(body):
        refuse("seal manifest hash does not match its content", "seal_tampered")
    snapshot = workdir / "seal" / "snapshot"
    for name, digest in manifest["files"].items():
        try:
            data = (snapshot / name).read_bytes()
        except OSError as error:
            refuse(f"sealed file {name} is unreadable: {error}", "seal_tampered")
        if sha_bytes(data) != digest:
            refuse(f"sealed file {name} no longer matches the seal", "seal_tampered")
    if manifest["source_sha256"] != ctx.source_sha256 or manifest["base_plan_sha256"] != ctx.old_plan.artifact["plan_sha256"]:
        refuse("seal belongs to another source or base plan", "seal_mismatch")
    if manifest["new_profile"] != profile_block(ctx.new_plan) or manifest["old_profile"] != profile_block(ctx.old_plan):
        refuse("seal profiles differ from the current old/new plans", "seal_mismatch")
    run = replay_old(ctx, snapshot)
    rebuilt = [chunk_entry(ctx, run, chunk) for chunk in ctx.old_plan.chunks]
    if rebuilt != manifest["chunks"]:
        refuse("offline revalidation of the sealed base no longer reproduces the seal", "seal_revalidation_mismatch")
    return manifest, run


# ##################################################################
# recovery plan
def recovery_targets(manifest: dict) -> list[str]:
    targets = [item["id"] for item in manifest["chunks"] if item["disposition"] in TARGET_DISPOSITIONS]
    if any(item["disposition"] == OLD_READY and item["id"] in targets for item in manifest["chunks"]):
        refuse("an old validated-ready chunk is among the recovery targets", "ready_chunk_targeted")
    return targets


def build_recovery_plan(ctx: Context, manifest: dict, run: DeltaRun) -> dict:
    targets = recovery_targets(manifest)
    unresolved = [item["id"] for item in manifest["chunks"] if item["disposition"] == OLD_UNRESOLVED]
    probe = RecoveryRun(ctx, Path("/nonexistent-recovery-probe"), run)
    first = None
    if targets:
        first_chunk = next(chunk for chunk in ctx.new_plan.chunks if chunk.id == targets[0])
        first = probe.prep(first_chunk, 1).request_sha256
    body = {
        "recovery_version": RECOVERY_VERSION,
        "seal_manifest_sha256": manifest["manifest_sha256"],
        "new_plan_sha256": ctx.new_plan.artifact["plan_sha256"],
        "old_plan_sha256": ctx.old_plan.artifact["plan_sha256"],
        "profile": NEW_PROFILE,
        "presence_penalty": PRESENCE_PENALTY,
        "targets": targets,
        "never_asked_old_ready": [item["id"] for item in manifest["chunks"] if item["disposition"] == OLD_READY],
        "old_unresolved": unresolved,
        "max_model_calls": len(targets),
        "first_target_request_sha256": first,
        "state_rule": "sequential: before each target the state holds the claims of every earlier old-validated-ready chunk and every earlier newly accepted chunk",
    }
    return body | {"recovery_plan_sha256": sha_json(body)}


# ##################################################################
# request contract
# fail closed before any request: the new payload must equal the old-profile payload for the same messages and schema plus exactly the presence penalty.
def check_request_contract(ctx: Context, payload: bytes) -> None:
    try:
        sent = json.loads(payload)
        messages = sent["messages"]
        schema = sent["response_format"]["json_schema"]["schema"]
    except (ValueError, KeyError, TypeError) as error:
        refuse(f"request payload is not a schema-constrained chat request: {error}", "request_contract")
    settings = ctx.new_plan.settings
    if sent.get("presence_penalty") != PRESENCE_PENALTY or sent.get("temperature") != settings.temperature or sent.get("seed") != settings.seed:
        refuse("request does not carry the recovery penalty/temperature/seed", "request_contract")
    if sent.get("max_tokens") != ctx.config.output_tokens:
        refuse("request output cap differs from the plan", "request_contract")
    _, old_payload = request_for(ctx.config.backend, messages, settings.temperature, ctx.config.output_tokens, schema, settings.seed, None)
    without = {key: value for key, value in sent.items() if key != "presence_penalty"}
    if without != json.loads(old_payload):
        refuse("request differs from the old-profile request by more than the presence penalty", "request_contract")


def require_presence_penalty_capability(models_document, model: str) -> None:
    """The server must advertise `capabilities.presence_penalty == presence-penalty-v1` for the configured model (GET /v1/models); anything else fails closed."""
    entries = models_document.get("data") if isinstance(models_document, dict) else None
    if not isinstance(entries, list):
        refuse("model listing is malformed; the presence-penalty capability is unproven", "capability_unproven")
    for entry in entries:
        if isinstance(entry, dict) and entry.get("id") == model:
            capabilities = entry.get("capabilities")
            if isinstance(capabilities, dict) and capabilities.get("presence_penalty") == CAPABILITY:
                return
            refuse(f"{model} does not advertise {CAPABILITY}", "capability_unproven")
    refuse(f"{model} is not in the model listing", "capability_unproven")


def fetch_models(config: ProofConfig, timeout: float = 30.0) -> dict:
    """The one metadata read the execute path makes (no inference)."""
    url = config.backend.url.rstrip("/")
    url = f"{url}/models" if url.endswith("/v1") else f"{url}/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# ##################################################################
# recovery run
# DeltaRun under the NEW plan whose journal holds only new-profile attempts.  Old validated-ready claims are fed into the state in chunk order (never into the journal and never re-asked); only targets are wanted.
class RecoveryRun(DeltaRun):
    def __init__(self, ctx: Context, out_dir: Path, old: DeltaRun) -> None:
        cast = build_cast_index(list(ctx.chapters), ctx.seed.get("registry", {}), ctx.seed.get("aliases", {}), None)
        super().__init__(list(ctx.chapters), ctx.config, ctx.new_plan, out_dir, ctx.count, cast, ctx.seed.get("registry"), ctx.seed.get("aliases"))
        self.old_ready = dict(old.ready)
        self.targets = frozenset(
            chunk.id for chunk in ctx.old_plan.chunks if disposition(old, chunk) in TARGET_DISPOSITIONS
        )
        self.index = {chunk.id: number for number, chunk in enumerate(ctx.new_plan.chunks)}
        self.applied_upto = 0

    def advance_to(self, chunk: DeltaChunk) -> None:
        stop = self.index[chunk.id]
        while self.applied_upto < stop:
            earlier = self.plan.chunks[self.applied_upto]
            self.applied_upto += 1
            if earlier.id in self.old_ready:
                self.state.apply(self.old_ready[earlier.id]["claims"])

    def prep(self, chunk: DeltaChunk, attempt: int) -> Prepared:
        self.advance_to(chunk)
        return super().prep(chunk, attempt)

    def replay(self, before_chunk: str | None = None) -> None:
        super().replay(before_chunk)
        stray = sorted(chunk_id for chunk_id, records in self.attempts.items() if records and chunk_id not in self.targets)
        if stray:
            refuse(f"recovery journal holds attempts for non-target chunks {stray}", "recovery_mismatch")

    def wants(self, chunk: DeltaChunk) -> bool:
        return chunk.id in self.targets and chunk.id not in self.ready and len(self.attempts[chunk.id]) < self.plan.settings.max_attempts


def recovery_dir(workdir: Path) -> Path:
    return workdir / "recovery"


def bind_recovery(workdir: Path, ctx: Context, recovery_plan: dict) -> None:
    out = recovery_dir(workdir)
    for name, data in (
        ("plan.json", json.dumps(ctx.new_plan.artifact, indent=2, sort_keys=True)),
        ("seed.json", canonical_json(ctx.seed)),
        ("recovery_plan.json", json.dumps(recovery_plan, indent=1, sort_keys=True)),
    ):
        path = out / name
        if path.is_file():
            if path.read_text(encoding="utf-8") != data:
                refuse(f"recovery {name} differs from this seal and plan; use a new work directory", "recovery_mismatch")
        else:
            write_atomic(path, data)


def recovery_state(ctx: Context, run: RecoveryRun, stop: str | None) -> dict:
    return {
        "new_plan_sha256": ctx.new_plan.artifact["plan_sha256"],
        "stop": stop,
        "targets": sorted(run.targets),
        "new_ready": sorted(run.ready),
        "target_status": {chunk_id: ("ready" if chunk_id in run.ready else run.chunk_state(ctx_chunk(ctx, chunk_id))) for chunk_id in sorted(run.targets)},
        "calls_total": run.calls_total(),
        "calls_this_invocation": run.calls_this_invocation,
    }


def ctx_chunk(ctx: Context, chunk_id: str) -> DeltaChunk:
    return next(chunk for chunk in ctx.new_plan.chunks if chunk.id == chunk_id)


def recover(
    ctx: Context,
    workdir: Path,
    transport: Transport | None,
    calibration: dict | None = None,
    models_document: dict | None = None,
    soft_s: float = SOFT_DEADLINE_S,
    hard_s: float = HARD_DEADLINE_S,
    clock: Callable[[], float] = time.monotonic,
    max_attempts: int = 1,
) -> dict:
    """Resume the recovery journal.  With transport=None nothing is sent (offline replay)."""
    manifest, old = verify_seal(ctx, workdir)
    unresolved = [item["id"] for item in manifest["chunks"] if item["disposition"] == OLD_UNRESOLVED]
    if unresolved:
        refuse(f"base chunks {unresolved} are neither validated, length-truncated nor missing", "old_unresolved")
    if transport is not None:
        measured = validate_calibration(calibration or {}, ctx.config)
        if measured["fixed_overhead_tokens"] != ctx.new_plan.fixed_overhead:
            refuse("calibration overhead differs from the plan", "calibration_overhead_mismatch")
        require_presence_penalty_capability(models_document, ctx.config.backend.model)
    plan = build_recovery_plan(ctx, manifest, old)
    out = recovery_dir(workdir)
    check_output_dir(out, list(ctx.chapters))
    new_plan = DeltaPlan(
        ctx.new_plan.source,
        ctx.new_plan.chunks,
        DeltaSettings(*[getattr(ctx.new_plan.settings, name) for name in ("target_tokens", "delta_tokens")], max_attempts, *[getattr(ctx.new_plan.settings, name) for name in ("temperature", "seed", "provider_grammar_profile", "compact_wire", "presence_penalty")]),
        ctx.new_plan.fixed_overhead,
        ctx.new_plan.artifact,
    )
    ctx = Context(ctx.config, ctx.chapters, ctx.source, ctx.source_sha256, ctx.count, ctx.stored_plan, ctx.seed, ctx.old_plan, new_plan)

    def guarded(url: str, payload: bytes, timeout: float) -> dict:
        check_request_contract(ctx, payload)
        return transport(url, payload, timeout)

    with run_lock(out):
        bind_recovery(workdir, ctx, plan)
        run = RecoveryRun(ctx, out, old)
        run.replay()
        started, stop = clock(), None
        try:
            for chunk in ctx.new_plan.chunks:
                if not run.wants(chunk):
                    continue
                stop = run.process(chunk, guarded if transport is not None else None, clock, started, soft_s, hard_s)
                write_atomic(out / "recovery_state.json", json.dumps(recovery_state(ctx, run, stop), indent=1, sort_keys=True))
                if stop:
                    break
        finally:
            write_atomic(out / "recovery_state.json", json.dumps(recovery_state(ctx, run, stop), indent=1, sort_keys=True))
        if stop == "fallback_route":
            refuse("server answered with a model other than the configured primary", "fallback_route")
    return recovery_state(ctx, run, stop)


# ##################################################################
# union verifier
# offline.  Replays the recovery journal (every saved raw response re-judged under rebuilt sequential state) and accounts for every chunk of the source exactly once.  A chunk is verified only as old_validated_ready (claims hash-identical to the seal) or new_accepted (replayed ready under the new plan); anything else, or any identity/accounting breach, makes the union not_ready.  This function never freezes or applies a cast.
def verify_union(ctx: Context, workdir: Path) -> dict:
    manifest, old = verify_seal(ctx, workdir)
    out = recovery_dir(workdir)
    run = RecoveryRun(ctx, out, old)
    if (out / "journal.jsonl").is_file():
        bound = json.loads((out / "recovery_plan.json").read_text(encoding="utf-8")) if (out / "recovery_plan.json").is_file() else None
        if bound is None or bound["seal_manifest_sha256"] != manifest["manifest_sha256"] or bound["new_plan_sha256"] != ctx.new_plan.artifact["plan_sha256"]:
            refuse("recovery journal is bound to another seal or plan", "recovery_mismatch")
        run.replay()
    rows, blockers, claims_all, pending_all = [], [], [], []
    expected_start = 0
    for entry in manifest["chunks"]:
        chunk = ctx_chunk(ctx, entry["id"])
        identity_ok = (
            chunk.start == entry["start"] == expected_start
            and chunk.end == entry["end"]
            and span_sha256(ctx, chunk) == entry["span_sha256"]
        )
        expected_start = chunk.end
        row = {"id": chunk.id, "start": chunk.start, "end": chunk.end, "span_sha256": entry["span_sha256"], "old_disposition": entry["disposition"], "identity_ok": identity_ok}
        validation = None
        if not identity_ok:
            row |= {"verified": False, "disposition": "identity_unresolved"}
        elif entry["disposition"] == OLD_READY:
            validation = old.ready[chunk.id]
            if validation_evidence(validation) != entry["old_validation"]:
                refuse(f"old claims of {chunk.id} no longer match the seal", "seal_revalidation_mismatch")
            row |= {"verified": True, "disposition": "verified_old_ready", "provenance": {"profile": OLD_PROFILE, "plan_sha256": ctx.old_plan.artifact["plan_sha256"], "request_sha256": entry["old_attempts"][-1]["request_sha256"], "response_sha256": entry["old_attempts"][-1]["response_sha256"]}}
        elif chunk.id in run.ready:
            validation = run.ready[chunk.id]
            record = run.attempts[chunk.id][-1]
            row |= {"verified": True, "disposition": "verified_new_accepted", "provenance": {"profile": NEW_PROFILE, "presence_penalty": PRESENCE_PENALTY, "plan_sha256": ctx.new_plan.artifact["plan_sha256"], "request_sha256": record["request_sha256"], "response_sha256": record["response_sha256"]}}
        else:
            last = run.attempts[chunk.id][-1]["status"] if run.attempts[chunk.id] else None
            row |= {"verified": False, "disposition": f"not_ready:{entry['disposition']}" + (f":new_{last}" if last else "")}
        if validation is not None:
            row["accounting"] = {"claims": len(validation["claims"]), "pending": len(validation["pending"]), "duplicates": len(validation["duplicates"]), "source_chars": chunk.end - chunk.start}
            claims_all.extend(validation["claims"])
            pending_all.extend(validation["pending"])
        else:
            blockers.append(row["id"])
        rows.append(row)
    ids = [claim["claim_id"] for claim in claims_all]
    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1}) if len(ids) != len(set(ids)) else []
    tiles = expected_start == len(ctx.source) == manifest["source_chars"]
    if duplicate_ids:
        blockers.append("duplicate_claim_ids")
    if not tiles:
        blockers.append("source_not_tiled")
    ready = not blockers and len(rows) == len(ctx.new_plan.chunks)
    reasons: dict[str, int] = {}
    for item in pending_all:
        reasons[item["pending_reason"]] = reasons.get(item["pending_reason"], 0) + 1
    return {
        "state": "ready" if ready else "not_ready",
        "blockers": blockers,
        "chunks": len(rows),
        "verified_chunks": sum(1 for row in rows if row["verified"]),
        "by_disposition": {name: sum(1 for row in rows if row["disposition"] == name) for name in sorted({row["disposition"] for row in rows})},
        "source": {"sha256": ctx.source_sha256, "chars": len(ctx.source), "tiled_exactly_once": tiles, "verified_chars": sum(row["end"] - row["start"] for row in rows if row["verified"])},
        "candidates": {"claims": len(claims_all), "pending": len(pending_all), "pending_reasons": reasons, "duplicate_claim_ids": duplicate_ids, "claims_sha256": sha_json(ids), "pending_sha256": sha_json([item["pending_id"] for item in pending_all])},
        "seal_manifest_sha256": manifest["manifest_sha256"],
        "new_plan_sha256": ctx.new_plan.artifact["plan_sha256"],
        "cast_status": "never frozen or applied by this module; every pending identity remains a freeze blocker",
        "freeze_allowed": False,
        "rows": rows,
    }


# ##################################################################
# command line
def caretaker_command(args, workdir: Path) -> str:
    parts = [
        "python -m src.wide_bio_recovery recover", f"{args.project}", f"--source {args.source}", f"--config {args.config}",
        f"--workdir {workdir}", f"--expect-source-sha256 {args.expect_source_sha256}", f"--expect-base-plan-sha256 {args.expect_base_plan_sha256}",
        "--calibration <valid calibration.json>", "--execute",
    ]
    return " ".join(parts)


def add_common(item: argparse.ArgumentParser, with_base: bool) -> None:
    item.add_argument("project", type=Path)
    item.add_argument("--source", type=Path, required=True)
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--workdir", type=Path, required=True)
    item.add_argument("--expect-source-sha256", required=True)
    item.add_argument("--expect-base-plan-sha256", required=True)
    if with_base:
        item.add_argument("--base", type=Path, required=True, help="base run directory (read-only)")


def context_for(args, workdir_files: Path | None) -> Context:
    if workdir_files is not None:
        base = workdir_files
        plan_json, seed_json = (base / "plan.json").read_bytes(), (base / "seed.json").read_bytes()
    else:
        plan_json, seed_json = (args.base / "plan.json").read_bytes(), (args.base / "seed.json").read_bytes()
    return load_context(args.project.resolve(), args.source, args.config, plan_json, seed_json, args.expect_source_sha256, args.expect_base_plan_sha256)


def main(argv: Sequence[str] | None = None, transport: Transport = chat_transport) -> int:
    parser = argparse.ArgumentParser(description="Offline seal/plan/verify and resumable recovery for a mixed-profile wide-bio run")
    sub = parser.add_subparsers(dest="command", required=True)
    add_common(sub.add_parser("seal"), True)
    add_common(sub.add_parser("plan"), False)
    add_common(sub.add_parser("verify"), False)
    rec = sub.add_parser("recover")
    add_common(rec, False)
    rec.add_argument("--calibration", type=Path, default=None)
    rec.add_argument("--max-attempts", type=int, default=1)
    rec.add_argument("--soft-deadline-s", type=float, default=SOFT_DEADLINE_S)
    rec.add_argument("--hard-deadline-s", type=float, default=HARD_DEADLINE_S)
    rec.add_argument("--execute", action="store_true", help="the only flag that can send requests; never used by the module's own tests")
    args = parser.parse_args(argv)
    try:
        workdir = args.workdir.resolve()
        ctx = context_for(args, None if args.command == "seal" else workdir / "seal" / "snapshot")
        if args.command == "seal":
            manifest = seal(ctx, args.base, workdir)
            print(json.dumps({"sealed": manifest["manifest_sha256"], "dispositions": manifest["dispositions"], "journal": manifest["journal"], "targets": len(manifest["targets"])}, sort_keys=True))
            return 0
        if args.command == "plan":
            manifest, old = verify_seal(ctx, workdir)
            plan = build_recovery_plan(ctx, manifest, old)
            write_atomic(workdir / "recovery_plan.json", json.dumps(plan, indent=1, sort_keys=True))
            print(json.dumps({key: plan[key] for key in ("recovery_plan_sha256", "seal_manifest_sha256", "new_plan_sha256", "max_model_calls", "first_target_request_sha256")} | {"targets": len(plan["targets"]), "old_unresolved": plan["old_unresolved"], "caretaker_command": caretaker_command(args, workdir)}, sort_keys=True))
            return 0 if not plan["old_unresolved"] else 3
        if args.command == "verify":
            result = verify_union(ctx, workdir)
            write_atomic(workdir / "union_verification.json", json.dumps(result, indent=1, sort_keys=True))
            print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
            return 0 if result["state"] == "ready" else 3
        calibration = models = None
        if args.execute:
            if args.calibration is None:
                refuse("--execute needs --calibration", "calibration_invalid")
            calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
            models = fetch_models(ctx.config)
        state = recover(ctx, workdir, transport if args.execute else None, calibration, models, args.soft_deadline_s, args.hard_deadline_s, max_attempts=args.max_attempts)
        print(json.dumps(state, sort_keys=True))
        return 0 if set(state["target_status"].values()) <= {"ready"} else 3
    except (ContractError, TokenizerRefusal) as error:
        print(json.dumps({"refused": getattr(error, "code", type(error).__name__), "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
