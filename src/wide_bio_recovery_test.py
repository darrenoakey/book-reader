"""Real-file tests for the mixed-profile recovery module.

No model, network or transport is involved: base and recovery evidence is written as real raw response/meta files and ingested through the existing DeltaRun content-addressed replay path (the same path a salvaged saved response takes), so the seal, plan, union verifier and request contract are exercised end to end."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest

from src.wide_bio import (
    ContractError,
    build_counter,
    canonical_json,
    load_proof_config,
    project_chapters,
    sha256_text,
    write_atomic,
)
from src.wide_bio_delta import (
    DeltaRun,
    DeltaSettings,
    Established,
    build_delta_plan,
    prepare,
)
from src.wide_bio_recovery import (
    MISSING,
    NEW_PROFILE,
    OLD_PROFILE,
    OLD_READY,
    OLD_TRUNCATED,
    PRESENCE_PENALTY,
    RecoveryRun,
    build_recovery_plan,
    check_request_contract,
    load_context,
    main,
    recover,
    recovery_dir,
    recovery_targets,
    require_presence_penalty_capability,
    seal,
    sha_json,
    verify_seal,
    verify_union,
)
from src.wide_bio_test import capture_file, make_config

TEXT = (
    "Ren was tall, with green eyes.\n\n"
    "Later Ren wore a red cloak and was twelve years old.\n\n"
    "Luna Starwaver, the Master, smiled.\n\n"
)
REGISTRY = {"ren": {"name": "Ren", "bio": "b", "look": "l", "facts": {"voice": ["calm"], "look": []}}}
ALIASES = {"ren": "ren"}
MODEL = "proof-model:1"
SOURCE_SHA = hashlib.sha256(TEXT.encode()).hexdigest()


def fact(name, ref, category, value, pid):
    return {"subject": {"name": name, "ref": ref}, "category": category, "value": value, "paragraph_id": pid}


READY0 = {"facts": [fact("Ren", "ren", "look", "tall, with green eyes", "000000")]}
READY1 = {"facts": [fact("Ren", "ren", "look", "a red cloak", "000001"), fact("Ren", "ren", "age", "twelve years old", "000001")]}
READY2 = {"facts": [fact("Luna Starwaver", "novel", "role", "the Master", "000002")]}


class World:
    def __init__(self, root: Path) -> None:
        self.root = root
        _path, digest = capture_file(root)
        self.config_path = make_config(root, digest, llm={"primary_style": "openai"})
        self.project = root / "project"
        (self.project / "chapters").mkdir(parents=True)
        (self.project / "chapters" / "01-one.txt").write_text(TEXT, encoding="utf-8")
        self.source = root / "source.txt"
        self.source.write_text(TEXT, encoding="utf-8")
        self.base = root / "base"
        self.work = root / "work"
        self.config = load_proof_config(self.config_path)
        self.chapters = project_chapters(self.project)
        self.count = build_counter(self.config)
        self.seed = {"registry": REGISTRY, "aliases": ALIASES}
        settings = DeltaSettings(1, 512, 1, 0.7, 1729, OLD_PROFILE)
        self.old_plan = build_delta_plan(TEXT, self.chapters, self.config, self.count, settings, 0, sha_json(self.seed))
        self.base_plan_sha = self.old_plan.artifact["plan_sha256"]

    def run(self, plan, out, state_claims=()):
        from src.cast_index import build_cast_index

        cast = build_cast_index(list(self.chapters), REGISTRY, ALIASES, None)
        run = DeltaRun(list(self.chapters), self.config, plan, out, self.count, cast, REGISTRY, ALIASES)
        return run

    def meta(self, prepared, raw, done="stop", eval_count=3):
        return json.dumps(
            {"model": MODEL, "done_reason": done, "prompt_eval_count": 7, "eval_count": eval_count, "request_sha256": prepared.request_sha256, "response_sha256": sha256_text(raw)},
            sort_keys=True,
        )

    def write_base(self, kinds: dict[str, str]) -> None:
        """kinds: chunk id -> 'ready' | 'truncated' (anything absent stays missing); journaled in order through the real DeltaRun."""
        self.base.mkdir(parents=True, exist_ok=True)
        write_atomic(self.base / "plan.json", json.dumps(self.old_plan.artifact, indent=2, sort_keys=True))
        write_atomic(self.base / "seed.json", canonical_json(self.seed))
        run = self.run(self.old_plan, self.base)
        run.replay()
        for chunk in self.old_plan.chunks:
            kind = kinds.get(chunk.id)
            if kind is None:
                continue
            prepared = run.prep(chunk, 1)
            raw = '{"facts": [{"subject"' if kind == "truncated" else json.dumps({"k0000": READY0, "k0001": READY1, "k0002": READY2}[chunk.id])
            r, m = run.raw_paths(chunk.id, 1)
            write_atomic(r, raw)
            write_atomic(m, self.meta(prepared, raw, "length" if kind == "truncated" else "stop", self.config.output_tokens if kind == "truncated" else 3))
            assert run.process(chunk, None, time.monotonic, time.monotonic(), 175, 240) is None

    def ctx(self):
        return load_context(self.project, self.source, self.config_path, (self.base / "plan.json").read_bytes(), (self.base / "seed.json").read_bytes(), SOURCE_SHA, self.base_plan_sha)

    def ctx_from_seal(self):
        snap = self.work / "seal" / "snapshot"
        return load_context(self.project, self.source, self.config_path, (snap / "plan.json").read_bytes(), (snap / "seed.json").read_bytes(), SOURCE_SHA, self.base_plan_sha)

    def write_recovery(self, ctx, chunk_ids: list[str]) -> None:
        """Real new-profile raw evidence for the chosen targets, journaled by RecoveryRun's own replay path."""
        manifest, old = verify_seal(ctx, self.work)
        out = recovery_dir(self.work)
        run = RecoveryRun(ctx, out, old)
        from src.wide_bio_recovery import bind_recovery

        bind_recovery(self.work, ctx, build_recovery_plan(ctx, manifest, old))
        run.replay()
        for chunk in ctx.new_plan.chunks:
            if chunk.id not in chunk_ids:
                continue
            prepared = run.prep(chunk, 1)
            check_request_contract(ctx, prepared.payload)
            raw = json.dumps({"k0001": READY1, "k0002": READY2, "k0000": READY0}[chunk.id])
            r, m = run.raw_paths(chunk.id, 1)
            write_atomic(r, raw)
            write_atomic(m, self.meta(prepared, raw))
            assert run.process(chunk, None, time.monotonic, time.monotonic(), 175, 240) is None


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def sealed(world: World, kinds=None):
    world.write_base(kinds or {"k0000": "ready", "k0001": "truncated"})
    return seal(world.ctx(), world.base, world.work)


def test_profile_regex_accepts_exact_provider_descriptor_and_old_descriptors() -> None:
    for name in (OLD_PROFILE, NEW_PROFILE, "upstream-json-blanks32+presence-penalty-v1"):
        assert DeltaSettings(1, 1, 1, 0.7, 1, name, True, PRESENCE_PENALTY if "+" in name else None).provider_grammar_profile == name
    for bad in ("+bad-profile", "Upper+case", "a+", "ab"):
        with pytest.raises(ContractError):
            DeltaSettings(1, 1, 1, 0.7, 1, bad)


def test_new_plan_differs_only_by_profile_and_penalty(world) -> None:
    sealed(world)
    ctx = world.ctx()
    assert ctx.old_plan.artifact == ctx.stored_plan and ctx.new_plan.artifact["plan_sha256"] != ctx.old_plan.artifact["plan_sha256"]
    assert ctx.new_plan.artifact["presence_penalty"] == PRESENCE_PENALTY and ctx.new_plan.artifact["provider_grammar_profile"] == NEW_PROFILE


def test_seal_dispositions_bind_hashes_and_never_write_base(world) -> None:
    world.write_base({"k0000": "ready", "k0001": "truncated"})
    before = {p.relative_to(world.base): p.read_bytes() for p in world.base.rglob("*") if p.is_file()}
    manifest = seal(world.ctx(), world.base, world.work)
    after = {p.relative_to(world.base): p.read_bytes() for p in world.base.rglob("*") if p.is_file()}
    assert before == after
    assert [c["disposition"] for c in manifest["chunks"]] == [OLD_READY, OLD_TRUNCATED, MISSING]
    assert manifest["targets"] == ["k0001", "k0002"] and manifest["source_sha256"] == SOURCE_SHA
    ready = manifest["chunks"][0]
    assert ready["old_validation"]["claims"] == 1 and len(ready["old_attempts"][0]["request_sha256"]) == 64
    assert ready["span_sha256"] == sha256_text(TEXT[ready["start"] : ready["end"]])
    assert manifest["old_profile"]["profile"] == OLD_PROFILE and manifest["new_profile"]["profile"] == NEW_PROFILE
    # old journal lines are copied into the sealed snapshot only, byte for byte, never into a new-profile journal
    assert (world.work / "seal" / "snapshot" / "journal.jsonl").read_bytes() == (world.base / "journal.jsonl").read_bytes()
    assert not (recovery_dir(world.work) / "journal.jsonl").exists()
    verify_seal(world.ctx_from_seal(), world.work)


def test_seal_snapshots_only_a_whole_line_prefix_of_a_growing_base(world) -> None:
    world.write_base({"k0000": "ready", "k0001": "truncated"})
    journal = world.base / "journal.jsonl"
    whole = journal.read_bytes()
    journal.write_bytes(whole + b'{"torn":')
    manifest = seal(world.ctx(), world.base, world.work)
    assert manifest["journal"]["records"] == 2
    assert (world.work / "seal" / "snapshot" / "journal.jsonl").read_bytes() == whole
    assert journal.read_bytes() == whole + b'{"torn":'


def test_seal_refusals(world) -> None:
    world.write_base({"k0000": "ready"})
    ctx = world.ctx()
    with pytest.raises(ContractError) as error:
        seal(ctx, world.base, world.base / "inside")
    assert error.value.code == "workdir_invalid"
    seal(ctx, world.base, world.work)
    with pytest.raises(ContractError) as error:
        seal(ctx, world.base, world.work)
    assert error.value.code == "seal_exists"
    plan_bytes, seed_bytes = (world.base / "plan.json").read_bytes(), (world.base / "seed.json").read_bytes()
    with pytest.raises(ContractError) as error:
        load_context(world.project, world.source, world.config_path, plan_bytes, seed_bytes, "0" * 64, world.base_plan_sha)
    assert error.value.code == "source_mismatch"
    with pytest.raises(ContractError) as error:
        load_context(world.project, world.source, world.config_path, plan_bytes, seed_bytes, SOURCE_SHA, "1" * 64)
    assert error.value.code == "base_plan_mismatch"
    penalised = json.loads(plan_bytes) | {"presence_penalty": 1.5}
    with pytest.raises(ContractError) as error:
        load_context(world.project, world.source, world.config_path, json.dumps(penalised).encode(), seed_bytes, SOURCE_SHA, world.base_plan_sha)
    assert error.value.code == "base_profile_mismatch"
    with pytest.raises(ContractError) as error:
        load_context(world.project, world.source, world.config_path, plan_bytes, json.dumps({"registry": {}, "aliases": {}}).encode(), SOURCE_SHA, world.base_plan_sha)
    assert error.value.code == "base_plan_mismatch"


def test_tampered_base_raw_is_refused_at_seal(world) -> None:
    world.write_base({"k0000": "ready"})
    raw = world.base / "raw" / "k0000.a1.response.txt"
    raw.write_text(raw.read_text() + " ", encoding="utf-8")
    with pytest.raises(ContractError) as error:
        seal(world.ctx(), world.base, world.work)
    assert error.value.code == "base_raw_mismatch"


def test_tampered_seal_is_refused(world) -> None:
    sealed(world)
    ctx = world.ctx_from_seal()
    snap = world.work / "seal" / "snapshot"
    raw = snap / "raw" / "k0000.a1.response.txt"
    original = raw.read_bytes()
    raw.write_bytes(original + b" ")
    with pytest.raises(ContractError) as error:
        verify_seal(ctx, world.work)
    assert error.value.code == "seal_tampered"
    raw.write_bytes(original)
    verify_seal(ctx, world.work)
    manifest_path = world.work / "seal" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][0]["disposition"] = MISSING
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError) as error:
        verify_seal(ctx, world.work)
    assert error.value.code == "seal_tampered"
    manifest["manifest_sha256"] = sha_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError) as error:
        verify_seal(ctx, world.work)
    assert error.value.code == "seal_revalidation_mismatch"


def test_recovery_targets_never_include_old_ready(world) -> None:
    manifest = sealed(world)
    assert recovery_targets(manifest) == ["k0001", "k0002"]
    forged = json.loads(json.dumps(manifest))
    forged["chunks"][0]["disposition"] = OLD_TRUNCATED
    assert recovery_targets(forged) == ["k0000", "k0001", "k0002"]  # a forged manifest is caught by verify_seal, never here
    ctx = world.ctx_from_seal()
    plan = build_recovery_plan(ctx, manifest, verify_seal(ctx, world.work)[1])
    assert plan["targets"] == ["k0001", "k0002"] and plan["never_asked_old_ready"] == ["k0000"] and plan["max_model_calls"] == 2
    assert len(plan["first_target_request_sha256"]) == 64


def test_request_contract_is_exactly_the_penalty(world) -> None:
    sealed(world)
    ctx = world.ctx_from_seal()
    _, old = verify_seal(ctx, world.work)
    run = RecoveryRun(ctx, recovery_dir(world.work), old)
    chunk = next(c for c in ctx.new_plan.chunks if c.id == "k0001")
    payload = run.prep(chunk, 1).payload
    check_request_contract(ctx, payload)
    body = json.loads(payload)
    assert body["presence_penalty"] == PRESENCE_PENALTY and body["temperature"] == 0.7 and body["seed"] == 1729
    # the new request equals the old-profile request for the same chunk and state plus the single penalty field
    old_chunk = next(c for c in ctx.old_plan.chunks if c.id == "k0001")
    state = Established()
    state.seed(REGISTRY, ALIASES)
    state.apply(old.ready["k0000"]["claims"])
    old_payload = json.loads(prepare(ctx.config, ctx.old_plan, old_chunk, state, ctx.count).payload)
    assert {k: v for k, v in body.items() if k != "presence_penalty"} == old_payload
    for mutate in (
        lambda b: b.pop("presence_penalty"),
        lambda b: b.update(presence_penalty=1.0),
        lambda b: b.update(temperature=0.8),
        lambda b: b.update(seed=1),
        lambda b: b.update(max_tokens=b["max_tokens"] + 1),
        lambda b: b["response_format"]["json_schema"].update(strict=False),
        lambda b: b.update(model="other:1"),
    ):
        tampered = json.loads(payload)
        mutate(tampered)
        with pytest.raises(ContractError) as error:
            check_request_contract(ctx, json.dumps(tampered).encode())
        assert error.value.code == "request_contract"


def test_capability_check_fails_closed() -> None:
    good = {"data": [{"id": MODEL, "capabilities": {"presence_penalty": "presence-penalty-v1"}}]}
    require_presence_penalty_capability(good, MODEL)
    for document in (
        {"data": [{"id": MODEL}]},
        {"data": [{"id": MODEL, "capabilities": {"presence_penalty": "other"}}]},
        {"data": [{"id": "x", "capabilities": {"presence_penalty": "presence-penalty-v1"}}]},
        {"data": "nope"},
        None,
        {},
    ):
        with pytest.raises(ContractError) as error:
            require_presence_penalty_capability(document, MODEL)
        assert error.value.code == "capability_unproven"


def test_execute_path_refuses_before_any_request_without_capability(world) -> None:
    sealed(world)
    ctx = world.ctx_from_seal()
    from src.wide_bio_test import calibration

    cal = calibration(ctx.config, ctx.config.tokenizer_sha256)
    cal["fixed_overhead_tokens"] = 0
    calls: list[str] = []

    def transport(url, payload, timeout):  # records any attempted send
        calls.append(url)
        raise AssertionError("no request may be sent")

    with pytest.raises(ContractError) as error:
        recover(ctx, world.work, transport, cal, {"data": [{"id": MODEL}]})
    assert error.value.code in {"capability_unproven", "calibration_insufficient", "calibration_overhead_mismatch"}
    assert calls == [] and not (recovery_dir(world.work) / "journal.jsonl").exists()


def test_recover_offline_resumes_and_union_needs_every_chunk(world) -> None:
    sealed(world)
    ctx = world.ctx_from_seal()
    not_ready = verify_union(ctx, world.work)
    assert not_ready["state"] == "not_ready" and not_ready["freeze_allowed"] is False
    assert not_ready["blockers"] == ["k0001", "k0002"] and not_ready["verified_chunks"] == 1
    world.write_recovery(ctx, ["k0001"])
    state = recover(ctx, world.work, None)
    assert state["new_ready"] == ["k0001"] and state["calls_total"] == 1 and state["stop"] == "offline"
    partial = verify_union(ctx, world.work)
    assert partial["state"] == "not_ready" and partial["blockers"] == ["k0002"]
    assert partial["by_disposition"] == {"not_ready:missing": 1, "verified_new_accepted": 1, "verified_old_ready": 1}
    world.write_recovery(ctx, ["k0002"])
    full = verify_union(ctx, world.work)
    assert full["state"] == "ready" and full["verified_chunks"] == 3 and full["blockers"] == []
    assert full["source"]["tiled_exactly_once"] and full["source"]["verified_chars"] == len(TEXT)
    rows = {row["id"]: row for row in full["rows"]}
    assert rows["k0000"]["provenance"]["profile"] == OLD_PROFILE and rows["k0001"]["provenance"]["profile"] == NEW_PROFILE
    assert rows["k0001"]["provenance"]["presence_penalty"] == PRESENCE_PENALTY and rows["k0002"]["accounting"]["claims"] == 1
    assert full["candidates"]["claims"] == 4 and full["freeze_allowed"] is False
    # the new journal holds only new-profile attempts for target chunks, never the old record
    journal = [json.loads(line) for line in (recovery_dir(world.work) / "journal.jsonl").read_text().splitlines()]
    assert [record["chunk_id"] for record in journal] == ["k0001", "k0002"]
    assert recover(ctx, world.work, None)["new_ready"] == ["k0001", "k0002"]


def test_recovery_state_is_chronological_union(world) -> None:
    sealed(world, {"k0000": "ready", "k0001": "truncated"})
    ctx = world.ctx_from_seal()
    _, old = verify_seal(ctx, world.work)
    run = RecoveryRun(ctx, recovery_dir(world.work), old)
    chunk = next(c for c in ctx.new_plan.chunks if c.id == "k0001")
    user = run.prep(chunk, 1).user
    assert "tall, with green eyes" in user  # old-ready claim of the earlier chunk is already established
    plain = Established()
    plain.seed(REGISTRY, ALIASES)
    assert "tall, with green eyes" not in prepare(ctx.config, ctx.new_plan, chunk, plain, ctx.count).user


def test_union_refuses_tampered_recovery_evidence_and_foreign_binding(world) -> None:
    sealed(world)
    ctx = world.ctx_from_seal()
    world.write_recovery(ctx, ["k0001"])
    raw = recovery_dir(world.work) / "raw" / "k0001.a1.response.txt"
    original = raw.read_bytes()
    raw.write_bytes(original + b" ")
    with pytest.raises(ContractError):
        verify_union(ctx, world.work)
    raw.write_bytes(original)
    verify_union(ctx, world.work)
    bound = recovery_dir(world.work) / "recovery_plan.json"
    data = json.loads(bound.read_text())
    data["seal_manifest_sha256"] = "0" * 64
    bound.write_text(json.dumps(data))
    with pytest.raises(ContractError) as error:
        verify_union(ctx, world.work)
    assert error.value.code == "recovery_mismatch"


def test_command_line_seal_plan_verify_are_offline(world, capsys) -> None:
    world.write_base({"k0000": "ready", "k0001": "truncated"})
    common = [str(world.project), "--source", str(world.source), "--config", str(world.config_path), "--workdir", str(world.work), "--expect-source-sha256", SOURCE_SHA, "--expect-base-plan-sha256", world.base_plan_sha]
    assert main(["seal", *common, "--base", str(world.base)]) == 0
    assert main(["plan", *common]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["targets"] == 2 and "--execute" in out["caretaker_command"] and out["old_unresolved"] == []
    assert main(["verify", *common]) == 3
    assert main(["recover", *common]) == 3  # offline replay only: no --execute, so nothing is sent
    assert main(["recover", *common, "--execute"]) == 2  # --execute needs a calibration; refused before any request
    shutil.rmtree(world.work / "seal" / "snapshot" / "raw")
    assert main(["verify", *common]) == 2
