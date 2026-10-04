"""Tests for the mixed-profile recovery module on ACTUAL captured production artifacts.

The success evidence is the real v8 production base: a hash-chained 12-record prefix of the actual 151-record sealed journal (src/testdata/wide_bio_v8_prefix: real plan.json, seed.json, journal and the real raw response/meta files the provider returned), the real 9 MB source, project chapters, proof config and tokenizer capture from the canonical checkout (read-only).  No provider output, token usage or salvage flag is invented.  Penalty-profile success evidence does not exist until the caretaker's pilot, so no recovery-ready state is fabricated: recovery coverage here is the real not_ready union.  Destructive tests clone the real bytes into namespaced temp directories and tamper with the clones; canonical inputs are never written."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.wide_bio import ContractError, chat_transport, sha256_text
from src.wide_bio_delta import DeltaSettings, Established, prepare
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

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "testdata" / "wide_bio_v8_prefix"
COMMON = Path(
    subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
).resolve().parent
SOURCE = COMMON / "incoming" / "weakest_beast_tamer.txt"
PROJECT = COMMON / "output" / "weakest_beast_tamer"
CONFIG = COMMON / "local" / "wide-bio-proof.toml"
CALIBRATION = COMMON / "local" / "wide-bio-proof" / "calibration-tf256k-valid.json"
SOURCE_SHA = "dc4d515d6ecd37d8c43b23618c0285b4a0dbfa296d861337a7cd70ce52b8070a"
BASE_PLAN_SHA = "1447a509670c89eaf95de96abbe0505c63973a7fdbf1a5565a19594d16b717b9"
NEW_PLAN_SHA = "e1f218d26304d1bbff019cee26f2544b9aa61755d3deb3d93a510e8a867fe293"
MODEL = "qwen3.6:35b-a3b"
JOURNAL = [json.loads(line) for line in (FIXTURE / "journal.jsonl").read_text().splitlines()]


def clone_base(dest: Path) -> Path:
    shutil.copytree(FIXTURE, dest)
    return dest


@pytest.fixture(scope="module")
def ctx():
    return load_context(PROJECT, SOURCE, CONFIG, (FIXTURE / "plan.json").read_bytes(), (FIXTURE / "seed.json").read_bytes(), SOURCE_SHA, BASE_PLAN_SHA)


@pytest.fixture(scope="module")
def sealed(tmp_path_factory, ctx):
    root = tmp_path_factory.mktemp("recovery-real")
    base = clone_base(root / "base")
    manifest = seal(ctx, base, root / "work")
    assert {p.relative_to(base): p.read_bytes() for p in base.rglob("*") if p.is_file()} == {p.relative_to(FIXTURE): p.read_bytes() for p in FIXTURE.rglob("*") if p.is_file()}
    return manifest, root / "work"


@pytest.fixture(scope="module")
def verified(sealed, ctx):
    return verify_seal(ctx, sealed[1])


@pytest.fixture
def work(sealed, tmp_path):
    """A namespaced clone of the real sealed work directory, safe to tamper with."""
    dest = tmp_path / "work"
    shutil.copytree(sealed[1], dest)
    return dest


def test_profile_regex_accepts_exact_provider_descriptor_and_old_descriptors() -> None:
    for name in (OLD_PROFILE, NEW_PROFILE, "upstream-json-blanks32+presence-penalty-v1"):
        assert DeltaSettings(1, 1, 1, 0.7, 1, name, True, PRESENCE_PENALTY if "+" in name else None).provider_grammar_profile == name
    for bad in ("+bad-profile", "Upper+case", "ab", "a b+c"):
        with pytest.raises(ContractError):
            DeltaSettings(1, 1, 1, 0.7, 1, bad)


def test_real_plan_fingerprints_are_preserved_and_new_plan_differs_only_by_profile_and_penalty(ctx) -> None:
    assert ctx.old_plan.artifact == ctx.stored_plan and ctx.old_plan.artifact["plan_sha256"] == BASE_PLAN_SHA
    assert ctx.new_plan.artifact["plan_sha256"] == NEW_PLAN_SHA
    old, new = ctx.old_plan.artifact, ctx.new_plan.artifact
    assert {key for key in old if old[key] != new.get(key)} | (set(new) - set(old)) == {"provider_grammar_profile", "presence_penalty", "plan_sha256"}
    assert (old["provider_grammar_profile"], new["provider_grammar_profile"], new["presence_penalty"]) == (OLD_PROFILE, NEW_PROFILE, PRESENCE_PENALTY)
    assert len(ctx.old_plan.chunks) == 246 and old["coverage"]["source_sha256"] == SOURCE_SHA


def test_seal_dispositions_come_from_the_actual_journal(sealed) -> None:
    manifest, work = sealed
    by_id = {record["chunk_id"]: record for record in JOURNAL}
    assert len(manifest["chunks"]) == 246 and manifest["journal"]["records"] == len(JOURNAL) == 12
    for entry in manifest["chunks"]:
        record = by_id.get(entry["id"])
        if record is None:
            assert entry["disposition"] == MISSING and entry["old_attempts"] == []
            continue
        assert entry["disposition"] == (OLD_READY if record["status"] == "ready" else OLD_TRUNCATED)
        attempt = entry["old_attempts"][0]
        assert (attempt["request_sha256"], attempt["response_sha256"], attempt["meta_sha256"], attempt["record_sha256"]) == tuple(record[k] for k in ("request_sha256", "response_sha256", "meta_sha256", "record_sha256"))
        if record["status"] == "ready":
            assert entry["old_validation"]["claims"] == record["counts"]["claims"] and entry["old_validation"]["pending"] == record["counts"]["pending"]
        assert entry["span_sha256"] == sha256_text(SOURCE.read_text(encoding="utf-8")[entry["start"] : entry["end"]])
    counts = manifest["dispositions"]
    assert counts == {OLD_READY: 8, OLD_TRUNCATED: 4, MISSING: 234, "old_unresolved": 0}
    assert manifest["targets"] == [e["id"] for e in manifest["chunks"] if e["disposition"] != OLD_READY]
    assert manifest["source_sha256"] == SOURCE_SHA and manifest["base_plan_sha256"] == BASE_PLAN_SHA
    assert manifest["new_profile"]["plan_sha256"] == NEW_PLAN_SHA and manifest["old_profile"]["profile"] == OLD_PROFILE
    assert (work / "seal" / "snapshot" / "journal.jsonl").read_bytes() == (FIXTURE / "journal.jsonl").read_bytes()
    assert not (recovery_dir(work) / "journal.jsonl").exists()


def test_seal_cuts_a_growing_base_at_the_last_whole_line(ctx, tmp_path) -> None:
    base = clone_base(tmp_path / "base")
    journal = base / "journal.jsonl"
    journal.write_bytes(journal.read_bytes() + b'{"seq":12,"torn":')
    manifest = seal(ctx, base, tmp_path / "work")
    assert manifest["journal"]["records"] == 12
    assert (tmp_path / "work" / "seal" / "snapshot" / "journal.jsonl").read_bytes() == (FIXTURE / "journal.jsonl").read_bytes()
    assert journal.read_bytes().endswith(b'"torn":')


def test_seal_and_context_refusals(ctx, sealed, tmp_path) -> None:
    base = clone_base(tmp_path / "base")
    with pytest.raises(ContractError) as error:
        seal(ctx, base, base / "inside")
    assert error.value.code == "workdir_invalid"
    with pytest.raises(ContractError) as error:
        seal(ctx, base, sealed[1])
    assert error.value.code == "seal_exists"
    plan_bytes, seed_bytes = (FIXTURE / "plan.json").read_bytes(), (FIXTURE / "seed.json").read_bytes()

    def build(plan=plan_bytes, seed=seed_bytes, source=SOURCE_SHA, plan_sha=BASE_PLAN_SHA):
        return load_context(PROJECT, SOURCE, CONFIG, plan, seed, source, plan_sha)

    for call, code in (
        (lambda: build(source="0" * 64), "source_mismatch"),
        (lambda: build(plan_sha="1" * 64), "base_plan_mismatch"),
        (lambda: build(plan=json.dumps(json.loads(plan_bytes) | {"presence_penalty": 1.5}).encode()), "base_profile_mismatch"),
        (lambda: build(plan=json.dumps(json.loads(plan_bytes) | {"provider_grammar_profile": NEW_PROFILE}).encode()), "base_profile_mismatch"),
    ):
        with pytest.raises(ContractError) as error:
            call()
        assert error.value.code == code


def test_tampered_real_base_is_refused_at_seal(ctx, tmp_path) -> None:
    base = clone_base(tmp_path / "base")
    raw = base / "raw" / "k0001.a1.response.txt"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ContractError) as error:
        seal(ctx, base, tmp_path / "work")
    assert error.value.code == "base_raw_mismatch"
    shutil.rmtree(tmp_path / "work", ignore_errors=True)
    base2 = clone_base(tmp_path / "base2")
    lines = (base2 / "journal.jsonl").read_text().splitlines()
    forged = json.loads(lines[1]) | {"status": "truncated"}
    lines[1] = json.dumps(forged)
    (base2 / "journal.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ContractError) as error:
        seal(ctx, base2, tmp_path / "work2")
    assert error.value.code == "base_journal_corrupt"


def test_tampered_seal_is_refused(ctx, work) -> None:
    snap = work / "seal" / "snapshot"
    raw = snap / "raw" / "k0001.a1.response.txt"
    original = raw.read_bytes()
    raw.write_bytes(original + b" ")
    with pytest.raises(ContractError) as error:
        verify_seal(ctx, work)
    assert error.value.code == "seal_tampered"
    raw.write_bytes(original)
    manifest_path = work / "seal" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks"][1]["disposition"] = MISSING
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError) as error:
        verify_seal(ctx, work)
    assert error.value.code == "seal_tampered"
    manifest["manifest_sha256"] = sha_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError) as error:
        verify_seal(ctx, work)
    assert error.value.code == "seal_revalidation_mismatch"


def test_targets_never_include_old_ready_and_plan_is_bound(ctx, sealed, verified) -> None:
    manifest, _ = sealed
    plan = build_recovery_plan(ctx, manifest, verified[1])
    ready = [e["id"] for e in manifest["chunks"] if e["disposition"] == OLD_READY]
    assert not set(plan["targets"]) & set(ready) and plan["never_asked_old_ready"] == ready
    assert plan["max_model_calls"] == len(plan["targets"]) == 238 and plan["old_unresolved"] == []
    assert plan["seal_manifest_sha256"] == manifest["manifest_sha256"] and plan["new_plan_sha256"] == NEW_PLAN_SHA
    forged = json.loads(json.dumps(manifest))
    forged["chunks"][1]["disposition"] = OLD_TRUNCATED
    assert "k0001" in recovery_targets(forged)  # caught by verify_seal's hash and revalidation, as the tamper test proves


def test_requests_equal_the_sealed_old_requests_plus_exactly_the_penalty(ctx, sealed, verified) -> None:
    manifest, work = sealed
    _, old = verified
    run = RecoveryRun(ctx, recovery_dir(work), old)
    entries = {entry["id"]: entry for entry in manifest["chunks"]}
    for chunk_id in ("k0000", "k0002", "k0011"):  # the real old_truncated chunks, including ones preceded by old-ready claims
        chunk = next(c for c in ctx.new_plan.chunks if c.id == chunk_id)
        payload = run.prep(chunk, 1).payload
        check_request_contract(ctx, payload)
        body = json.loads(payload)
        assert body["presence_penalty"] == PRESENCE_PENALTY and body["temperature"] == 0.7 and body["seed"] == 1729
        stripped = json.dumps({k: v for k, v in body.items() if k != "presence_penalty"}).encode()
        # the same request without the penalty is byte-for-byte the real request the provider answered in the base run
        assert hashlib.sha256(stripped).hexdigest() == entries[chunk_id]["old_attempts"][0]["request_sha256"]
    state = Established()
    state.seed(ctx.seed["registry"], ctx.seed["aliases"])
    chunk = next(c for c in ctx.new_plan.chunks if c.id == "k0002")
    assert run.prep(chunk, 1).user != prepare(ctx.config, ctx.new_plan, chunk, state, ctx.count).user  # old-ready claims are in the sequential state
    payload = run.prep(chunk, 1).payload
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
    require_presence_penalty_capability({"data": [{"id": MODEL, "capabilities": {"presence_penalty": "presence-penalty-v1"}}]}, MODEL)
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


def test_execute_refuses_before_any_request_and_attempts_are_exactly_one(ctx, work) -> None:
    import inspect

    calibration = json.loads(CALIBRATION.read_text())
    assert "max_attempts" not in inspect.signature(recover).parameters
    for models in ({"data": [{"id": MODEL}]}, None):
        with pytest.raises(ContractError) as error:
            recover(ctx, work, chat_transport, calibration, models)  # real transport; refused before it can be called
        assert error.value.code == "capability_unproven"
    assert not (recovery_dir(work) / "journal.jsonl").exists()
    loose = replace(ctx, new_plan=replace(ctx.new_plan, settings=replace(ctx.new_plan.settings, max_attempts=2)))
    with pytest.raises(ContractError) as error:
        recover(loose, work, None)
    assert error.value.code == "attempts_not_one"


def test_real_union_is_not_ready_and_never_frees_a_freeze(ctx, sealed, work) -> None:
    manifest, _ = sealed
    state = recover(ctx, work, None)  # offline replay: nothing is sent
    assert state["stop"] == "offline" and state["new_ready"] == [] and state["calls_total"] == 0
    union = verify_union(ctx, work)
    assert union["state"] == "not_ready" and union["source_coverage_ready"] is False and union["freeze_allowed"] is False
    assert union["full_candidate_accounting"].startswith("not_verified")
    assert union["verified_chunks"] == 8 and union["chunks"] == 246 and len(union["blockers"]) == 238
    assert union["by_disposition"] == {"not_ready:missing": 234, "not_ready:old_truncated": 4, "verified_old_ready": 8}
    old_ready = [e for e in manifest["chunks"] if e["disposition"] == OLD_READY]
    assert union["candidates"]["claims"] == sum(e["old_validation"]["claims"] for e in old_ready)
    assert union["candidates"]["pending"] == sum(e["old_validation"]["pending"] for e in old_ready)
    assert union["source"]["tiled_exactly_once"] and union["source"]["verified_chars"] == sum(e["end"] - e["start"] for e in old_ready)
    rows = {row["id"]: row for row in union["rows"]}
    assert rows["k0001"]["provenance"]["profile"] == OLD_PROFILE and rows["k0001"]["provenance"]["request_sha256"]
    assert rows["k0000"]["verified"] is False and rows["k0000"]["disposition"] == "not_ready:old_truncated"


def test_union_refuses_a_foreign_recovery_binding_and_stray_evidence(ctx, work) -> None:
    recover(ctx, work, None)
    out = recovery_dir(work)
    (out / "journal.jsonl").write_bytes((work / "seal" / "snapshot" / "journal.jsonl").read_bytes())  # old records must never sit in the new-profile journal
    with pytest.raises(ContractError):
        verify_union(ctx, work)
    (out / "journal.jsonl").unlink()
    bound = out / "recovery_plan.json"
    data = json.loads(bound.read_text())
    (out / "journal.jsonl").write_bytes(b"")
    data["seal_manifest_sha256"] = "0" * 64
    bound.write_text(json.dumps(data))
    with pytest.raises(ContractError) as error:
        verify_union(ctx, work)
    assert error.value.code == "recovery_mismatch"


def test_command_line_is_offline_and_fail_closed(work, capsys) -> None:
    common = [str(PROJECT), "--source", str(SOURCE), "--config", str(CONFIG), "--workdir", str(work), "--expect-source-sha256", SOURCE_SHA, "--expect-base-plan-sha256", BASE_PLAN_SHA]
    assert main(["plan", *common]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["targets"] == 238 and out["new_plan_sha256"] == NEW_PLAN_SHA and "--execute" in out["caretaker_command"] and "--max-attempts" not in out["caretaker_command"]
    assert main(["recover", *common, "--execute"]) == 2  # needs a calibration
    with pytest.raises(SystemExit):
        main(["recover", *common, "--max-attempts", "2"])
