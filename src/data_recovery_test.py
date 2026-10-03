"""Real-filesystem behavior tests for the resilient data contract (no mocks)."""

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from src.audio_synth import plan_chapter
from src.cast_freeze import ANCHOR_IDS, MANIFEST_NAME, PROGRESS_NAME, prepare_cast
from src.character_analysis import (
    apply_dedup_groups,
    merge_character_info,
    parse_json_response_strict,
)
from src.data_recovery import (
    DataIssue,
    OperationalError,
    RecoveryLedger,
    load_json_store,
)
from src.epub_extract import get_output_dir
from src.hour_runner import chapter_order

REAL_CHAPTERS = Path("/Users/darrenoakey/src/book-reader/output/weakest_beast_tamer/chapters")


def bad_json_ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
    """Native endpoint returning an unusable payload for every request."""
    return '{"classifications": [ {oops'


def make_project(chapters: dict[str, str]) -> tuple[Path, Path]:
    source = Path(f"/tmp/drc_{uuid.uuid4().hex}.txt")
    source.write_text("source", encoding="utf-8")
    project = get_output_dir(source)
    (project / "chapters").mkdir(parents=True)
    (project / "chapters" / "00-intro.txt").write_text("Book by Tester, narrated by Narrator", encoding="utf-8")
    for name, text in chapters.items():
        (project / "chapters" / name).write_text(text, encoding="utf-8")
    (project / "characters.json").write_text(
        json.dumps({a: {"name": a, "bio": "", "look": ""} for a in ANCHOR_IDS}), encoding="utf-8"
    )
    return source, project


def real_chapter(position: int) -> str:
    ordered = sorted((p for p in REAL_CHAPTERS.glob("*.txt") if not p.name.startswith("00-")), key=chapter_order)
    return ordered[position - 1].read_text(encoding="utf-8")


def test_original49_and_62_with_malformed_neighbours_continue_and_block_freeze() -> None:
    assert REAL_CHAPTERS.is_dir(), "mandatory real weakest-beast-tamer chapters are unavailable"
    source, project = make_project(
        {
            "01-real49.txt": real_chapter(49),
            "02-empty.txt": "   \n",
            "03-plain.txt": "it rained all day. nothing happened at all.",
            "04-real62.txt": real_chapter(62),
            "05-José.txt": "José spoke to Ren. Ren listened to José carefully.",
        }
    )
    try:
        result = prepare_cast(source, ask=bad_json_ask)
        assert result["status"] == "blocked" and result["pending"] >= 1
        assert not (project / MANIFEST_NAME).exists()
        progress = json.loads((project / PROGRESS_NAME).read_text())
        assert progress["next_chapter"] == 5 and progress["semantic_coverage"]["next_chapter"] == 5
        rows = RecoveryLedger(project).entries()
        quarantine_rows = [r for r in rows if r["severity"] == "quarantine"]
        quarantined = {r["evidence"]["chapters"][0] for r in quarantine_rows}
        # An empty chapter is a valid zero-coverage attestation with its exact hash, never a quarantine or freeze veto.
        assert {"01-real49.txt", "04-real62.txt", "05-José.txt"} <= quarantined and "02-empty.txt" not in quarantined
        assert progress["semantic_coverage"]["completed_batches"][1]["empty_chapters"] == {
            "02-empty.txt": hashlib.sha256(b"   \n").hexdigest()
        }
        assert all(r["checkpoint"]["next_chapter"] >= 1 and r["evidence"]["error_type"] for r in quarantine_rows)
        # restart: no raise, no duplicate rows, still blocked
        again = prepare_cast(source, ask=bad_json_ask)
        assert again["status"] == "blocked" and again["quarantined_chapters"] == result["quarantined_chapters"]
        assert [r for r in RecoveryLedger(project).entries() if r["severity"] == "quarantine"] == quarantine_rows
    finally:
        shutil.rmtree(project, ignore_errors=True)
        source.unlink(missing_ok=True)


def test_operational_store_failures_fail_closed() -> None:
    source, project = make_project({"01-a.txt": "Plain text here."})
    try:
        (project / "characters.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(OperationalError) as caught:
            prepare_cast(source, ask=bad_json_ask)
        assert caught.value.status == "store_unreadable"
        (project / "characters.json").unlink()
        with pytest.raises(OperationalError):
            prepare_cast(source, ask=bad_json_ask)
        with pytest.raises(OperationalError):
            load_json_store(project / "absent.json", "thing")
    finally:
        shutil.rmtree(project, ignore_errors=True)
        source.unlink(missing_ok=True)


def test_analysis_bad_json_is_typed_not_empty_and_dedup_never_guesses() -> None:
    for payload in ("", "no json at all", "[1, 2]", "{broken"):
        with pytest.raises(DataIssue):
            parse_json_response_strict(payload)
    assert parse_json_response_strict('```json\n{"characters": {}}\n```') == {"characters": {}}
    project = Path(f"/tmp/drc_{uuid.uuid4().hex}")
    try:
        ledger = RecoveryLedger(project)
        merged = merge_character_info(
            [{"characters": {"a": {"name": "A", "voice": "v", "look": "l"}, "b": "bad", "c": {"name": 3}}}, "junk"],
            ledger,
        )
        assert set(merged) == {"a"}
        assert len(ledger.entries()) == 3
        chars = {i: {"name": i, "bio": "x", "look": ""} for i in ("a", "b", "c")}
        out = apply_dedup_groups(chars, [["a", "b"], ["b", "c"], "bad", ["zzz", "a"]], ledger)
        assert set(out) == {"a", "c"} or set(out) == {"b", "c"}
        codes = {r["code"] for r in ledger.entries()}
        assert {"dedup_group_ambiguous", "dedup_group_malformed", "dedup_unknown_ids"} <= codes
    finally:
        shutil.rmtree(project, ignore_errors=True)


def test_audio_unknown_speaker_is_not_substituted_and_malformed_line_is_quarantined(tmp_path: Path) -> None:
    script = tmp_path / "01-x.jsonl"
    script.write_text(
        '{"narrator": "Hello there."}\n{"ghost": "boo"}\nnot json\n{"narrator": ""}\n[1]\n', encoding="utf-8"
    )
    audio = tmp_path / "audio"
    audio.mkdir()
    ledger = RecoveryLedger(tmp_path)
    _, _paths, _, meta = plan_chapter(script, audio, tmp_path / "voices", {"narrator"}, ledger)
    assert [m["text"] for m in meta] == ["Hello there."]
    codes = sorted(r["code"] for r in ledger.entries())
    assert codes == [
        "script_line_malformed",
        "script_line_text_unusable",
        "script_line_unparseable",
        "script_speaker_unknown",
    ]


def test_ledger_record_is_total_over_hostile_evidence_then_next_valid_and_pending_persist(tmp_path: Path) -> None:
    cyclic: dict = {"name": "loop"}
    cyclic["self"] = cyclic
    deep: object = "leaf"
    for _ in range(50):
        deep = [deep]
    raw = b"\xff\xfe\x00binary" * 1000
    hostile = {
        "cycle": cyclic,
        "surrogate": "bad\ud800text",
        "nan": float("nan"),
        "inf": [float("inf"), float("-inf")],
        "raw": raw,
        "deep": deep,
        "object": object(),
        "set": {1, 2},
        1: "non-string key",
        "source": "01-a.txt",
        "sha256": "aaa",
    }
    ledger = RecoveryLedger(tmp_path)
    row = ledger.record(
        "cast", "01-a.txt", "bad", "malformed", severity="quarantine", evidence=hostile, checkpoint=hostile
    )
    persisted = ledger.entries()
    assert len(persisted) == 1
    json.dumps(persisted[0], allow_nan=False)
    evidence = persisted[0]["evidence"]
    assert evidence["cycle"]["self"] == {"__cycle__": "dict"}
    assert evidence["raw"]["sha256"] == __import__("hashlib").sha256(raw).hexdigest()
    assert evidence["raw"]["__bytes__"] == len(raw) and evidence["raw"]["truncated"] is True
    assert evidence["nan"] == {"__float__": "nan"}
    assert evidence["surrogate"]["chars"] == 8 and len(row["evidence_sha256"]) == 64
    # Full payload hash tracks content beyond the bounded rendering.
    other = dict(hostile, raw=raw + b"x")
    assert (
        ledger.record("cast", "01-a.txt", "bad", "malformed", evidence=other)["evidence_sha256"]
        != row["evidence_sha256"]
    )
    # Same stage/item/code but different exact source scope/hash is a distinct record; exact repeat is not.
    ledger.record("cast", "01-a.txt", "bad", "malformed", evidence={"source": "01-a.txt", "sha256": "bbb"})
    ledger.record("cast", "01-a.txt", "bad", "malformed", evidence={"source": "01-a.txt", "sha256": "bbb"})
    assert len(ledger.entries()) == 2
    # The next valid item and a PENDING cursor record persist across a fresh ledger instance.
    ledger.record("cast", "02-b.txt", "pending_batch", "waiting", severity="pending", checkpoint={"cursor": 2})
    reloaded = RecoveryLedger(tmp_path)
    rows = reloaded.entries()
    assert [r["item"] for r in rows] == ["01-a.txt", "01-a.txt", "02-b.txt"]
    assert rows[-1]["severity"] == "pending" and rows[-1]["checkpoint"] == {"cursor": 2}
    reloaded.record("cast", "02-b.txt", "pending_batch", "waiting", severity="pending", checkpoint={"cursor": 2})
    assert len(reloaded.entries()) == 3


def test_ledger_write_failure_stays_operational(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(OperationalError) as caught:
        RecoveryLedger(blocker / "sub").record("s", "i", "c", "m", evidence={"x": float("nan")})
    assert caught.value.status == "recovery_ledger_unwritable"


def test_pending_rows_resolve_durably_without_rewriting_history(tmp_path: Path) -> None:
    ledger = RecoveryLedger(tmp_path)
    scope = {"source": ["01-a.txt"], "source_hash": {"01-a.txt": "h1"}}
    ledger.record("cast", "chapters 0-0:x", "verifier_uncertain", "pending", severity="pending", evidence=scope)
    ledger.record("cast", "chapters 1-1:y", "verifier_uncertain", "pending", severity="pending", evidence=scope)
    before = (tmp_path / "data_recovery.jsonl").read_text()
    (first, second) = ledger.open_pending("cast")
    ledger.resolve(first, "replayed_exact_scope_clean")
    after = (tmp_path / "data_recovery.jsonl").read_text()
    assert after.startswith(before) and len(after.splitlines()) == 3
    fresh = RecoveryLedger(tmp_path)
    assert [r["item"] for r in fresh.open_pending("cast")] == [second["item"]]
    assert fresh.resolve(first, "replayed_exact_scope_clean")["severity"] == "resolved"
    assert len((tmp_path / "data_recovery.jsonl").read_text().splitlines()) == 3


def test_replay_resolves_only_exact_scope_and_leaves_cursor(tmp_path: Path) -> None:
    from src.cast_freeze import file_digest, replay_pending, semantic_coverage

    chapter = tmp_path / "01-a.txt"
    chapter.write_text("it rained. nothing happened.", encoding="utf-8")
    progress = {"registry": {}, "aliases": {}, "next_chapter": 1}
    ledger = RecoveryLedger(tmp_path)
    good = {"source": ["01-a.txt"], "source_hash": {"01-a.txt": file_digest(chapter)}}
    stale = {"source": ["01-a.txt"], "source_hash": {"01-a.txt": "0" * 64}}
    ledger.record("cast", "chapters 0-0:ok", "verifier_uncertain", "pending", severity="pending", evidence=good)
    ledger.record("cast", "chapters 0-0:old", "verifier_uncertain", "pending", severity="pending", evidence=stale)
    summary = replay_pending(tmp_path, [chapter], progress, set(), chapter.read_text(), ledger, bad_json_ask)
    assert summary["resolved"] == 1 and summary["open"] == 1
    assert [r["item"] for r in ledger.open_pending("cast")] == ["chapters 0-0:old"]
    assert progress["next_chapter"] == 1
    assert len(semantic_coverage(progress)["resolved_pending"]) == 1


def test_replay_resolution_records_the_model_bindings_it_relied_on(tmp_path: Path) -> None:
    import hashlib

    from src.cast_freeze import (
        SCOPED_AUDIT_NAME,
        file_digest,
        immutable_evidence_units,
        replay_pending,
        semantic_coverage,
    )

    chapter = tmp_path / "01-a.txt"
    chapter.write_text("Young Ren smiled. Young bowed.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    quote = units[1]["quote"]
    record = {
        "chapter_sha256": units[1]["chapter_sha256"],
        "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
        "label": "Young",
        "span_start": 0,
        "canonical": "young_ren",
        "decision": "alias",
        "confidence": 0.9,
        "reason": "context",
        "proof": {"literal": "shared_name_word"},
    }
    (tmp_path / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": [record]}))
    progress = {"registry": {"young_ren": {"name": "Young Ren"}}, "aliases": {}, "next_chapter": 1}
    ledger = RecoveryLedger(tmp_path)
    scope = {"source": ["01-a.txt"], "source_hash": {"01-a.txt": file_digest(chapter)}}
    ledger.record(
        "cast", "chapters 0-0:young", "pending_uncertain_living", "pending", severity="pending", evidence=scope
    )

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        out = []
        for option in response_schema["properties"]["classifications"]["items"]["oneOf"]:
            branches = option.get("oneOf", [option])
            cid = branches[0]["properties"]["candidate_id"]["enum"][0]
            row = next(line for line in prompt.splitlines() if line.startswith(cid + " label="))
            statuses = {st: b["properties"] for b in branches for st in b["properties"]["status"]["enum"]}
            status = next(iter(statuses)) if len(branches) == 1 else "non_character"
            out.append(
                {
                    "candidate_id": cid,
                    "status": status,
                    "identity": statuses[status]["identity"]["enum"][0],
                    "evidence_unit_ids": [row.split("witnesses: [")[1].split("]")[0]],
                }
            )
        return json.dumps({"classifications": out})

    summary = replay_pending(tmp_path, [chapter], progress, set(), chapter.read_text(), ledger, ask)
    assert summary["resolved"] == 1
    (entry,) = semantic_coverage(progress)["resolved_pending"]
    (binding,) = entry["model_bindings"]
    assert binding["label"] == "Young" and binding["canonical"] == "young_ren" and binding["proof"]
    resolution = [r for r in ledger.entries() if r["severity"] == "resolved"]
    assert resolution[0]["evidence"]["model_bindings"] == entry["model_bindings"]


def test_quarantine_archives_raw_bytes_empty_code_and_invalid_unicode(tmp_path: Path) -> None:
    from src.cast_freeze import empty_chapter_attestations, recoverable_batches

    empty = tmp_path / "01-empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    bad = tmp_path / "02-bad.txt"
    bad.write_bytes(b"Ren said \xe3\x81 hello")
    progress = {"registry": {}, "aliases": {}}
    ledger = RecoveryLedger(tmp_path)
    units = []
    for index in range(2):
        units += list(
            recoverable_batches(tmp_path, [empty, bad], index, progress, set(), "", ledger, bad_json_ask, None)
        )
    assert units[0]["quarantine"] is None and units[0]["units"] == []
    assert empty_chapter_attestations([empty]) == {"01-empty.txt": hashlib.sha256(b"  \n").hexdigest()}
    assert units[1]["quarantine"] == ["02-bad.txt"]
    rows = [r for r in ledger.entries() if r["stage"] == "cast"]
    assert [r["code"] for r in rows] == ["source_not_utf8"]
    archive = [json.loads(line) for line in (tmp_path / "cast_preparation_rejections.jsonl").read_text().splitlines()]
    assert (
        len(archive) == 1
        and archive[0]["raw_chapter_sha256"]["02-bad.txt"] == rows[0]["evidence"]["raw_chapter_sha256"]["02-bad.txt"]
    )
    assert rows[0]["evidence"]["archive_line"] == 0
    assert archive[0]["evidence"]["invalid_bytes_hex"] == "e381"


def test_programming_errors_are_not_quarantined_and_oversized_chapter_windows(tmp_path: Path) -> None:
    from src.cast_freeze import (
        PREPARATION_PROMPT_MAX_CHARS,
        recoverable_batches,
        unit_windows,
    )

    chapter = tmp_path / "01-a.txt"
    chapter.write_text("Ren spoke.", encoding="utf-8")

    def broken_ask(*args, **kwargs):
        raise KeyError("programmer bug")

    progress = {"registry": {}, "aliases": {}}
    with pytest.raises(KeyError):
        list(
            recoverable_batches(tmp_path, [chapter], 0, progress, set(), "", RecoveryLedger(tmp_path), broken_ask, None)
        )
    sentence = "Ren walked on and on through the quiet hall. "
    units = [
        {"id": f"c00s{i:05d}", "chapter": "x", "chapter_sha256": "h", "quote": sentence}
        for i in range(PREPARATION_PROMPT_MAX_CHARS // len(sentence))
    ]
    windows = unit_windows(units, {}, {})
    assert len(windows) > 1 and [u for w in windows for u in w] == units


def legacy_quarantine_project(tmp_path: Path, texts: dict[str, bytes]) -> tuple[list[Path], dict, RecoveryLedger]:
    """Real files plus a legacy-shaped quarantine ledger row and quarantined coverage entry per chapter."""
    from src.cast_freeze import file_digest, semantic_coverage

    chapters = []
    for name, data in texts.items():
        path = tmp_path / name
        path.write_bytes(data)
        chapters.append(path)
    progress = {"registry": {}, "aliases": {}, "next_chapter": len(chapters), "completed_batches": []}
    coverage = semantic_coverage(progress)
    ledger = RecoveryLedger(tmp_path)
    for index, path in enumerate(chapters):
        digest = file_digest(path)
        entry = {"start": index, "end": index + 1, "chapter_sha256": {path.name: digest}}
        progress["completed_batches"].append({**entry, "quarantined": True})
        coverage["completed_batches"].append({**entry, "ledger_sha256": "x", "quarantined": True})
        # legacy shape: no source/source_hash, only chapters/raw_chapter_sha256 and a severity of quarantine
        ledger.record(
            "cast",
            f"chapters {index}-{index}",
            "cast_evidence_rejected",
            "legacy rejection",
            severity="quarantine",
            evidence={"chapters": [path.name], "raw_chapter_sha256": {path.name: digest}, "hash": digest},
        )
    coverage["next_chapter"] = len(chapters)
    (tmp_path / "cast_preparation_progress.json").write_text("{}", encoding="utf-8")
    return chapters, progress, ledger


def test_legacy_quarantines_revalidate_by_exact_hash_with_append_only_history(tmp_path: Path) -> None:
    from src.cast_freeze import (
        empty_chapter_names,
        quarantined_chapter_names,
        revalidate_quarantines,
    )

    chapters, progress, ledger = legacy_quarantine_project(
        tmp_path, {"01-plain.txt": b"it rained all day.", "02-empty.txt": b"  \n", "03-bad.txt": b"Ren said \xe3\x81 x"}
    )
    before = (tmp_path / "data_recovery.jsonl").read_text()
    summary = revalidate_quarantines(tmp_path, chapters, progress, set(), "", ledger, bad_json_ask)
    assert summary["resolved"] == 2 and summary["open"] == 1 and summary["attempts"] == 3
    # clean + empty chapters are attested; invalid UTF-8 stays quarantined with exact evidence, never blanket-accepted
    assert quarantined_chapter_names(progress) == ["03-bad.txt"]
    assert empty_chapter_names(progress) == {"02-empty.txt": hashlib.sha256(b"  \n").hexdigest()}
    after = (tmp_path / "data_recovery.jsonl").read_text()
    assert after.startswith(before)  # ledger history is append-only
    rows = [json.loads(line) for line in after.splitlines()]
    assert [r["severity"] for r in rows[3:]].count("resolved") == 2
    assert {r["code"] for r in rows if r["severity"] == "quarantine"} == {
        "cast_evidence_rejected",
        "source_not_utf8",  # the fresh exact evidence of the still-undecodable chapter is appended, not overwritten
    }
    assert len(progress["semantic_coverage"]["superseded_quarantines"]) == 2
    assert chapters[2].read_bytes() == b"Ren said \xe3\x81 x" and progress["next_chapter"] == 3
    # Bounded: a persistent failure stops after the attempt limit, and resolved rows are never retried.
    for _ in range(5):
        revalidate_quarantines(tmp_path, chapters, progress, set(), "", RecoveryLedger(tmp_path), bad_json_ask)
    final = RecoveryLedger(tmp_path)
    open_rows = final.open_quarantined("cast")
    assert {r["code"] for r in open_rows} == {"cast_evidence_rejected", "source_not_utf8"}
    assert [len(final.attempts(r)) for r in open_rows] == [3, 3]
    assert len([r for r in final.entries() if r["severity"] == "resolved"]) == 2


def test_changed_source_bytes_are_stale_and_never_retried(tmp_path: Path) -> None:
    from src.cast_freeze import quarantined_chapter_names, revalidate_quarantines

    chapters, progress, ledger = legacy_quarantine_project(tmp_path, {"01-plain.txt": b"it rained all day."})
    chapters[0].write_bytes(b"it rained all night.")
    summary = revalidate_quarantines(tmp_path, chapters, progress, set(), "", ledger, bad_json_ask)
    assert summary["stale"] == 1 and summary["attempts"] == 0 and summary["resolved"] == 0
    assert quarantined_chapter_names(progress) == ["01-plain.txt"]


def test_revalidation_call_limit_defers_remaining_rows(tmp_path: Path) -> None:
    from src.cast_freeze import revalidate_quarantines

    chapters, progress, ledger = legacy_quarantine_project(
        tmp_path, {f"0{i}-plain.txt": f"it rained {i}.".encode() for i in range(1, 4)}
    )
    first = revalidate_quarantines(tmp_path, chapters, progress, set(), "", ledger, bad_json_ask, limit=1)
    assert first["resolved"] == 1 and first["deferred"] == 2
    second = revalidate_quarantines(tmp_path, chapters, progress, set(), "", RecoveryLedger(tmp_path), bad_json_ask)
    assert second["resolved"] == 2 and not RecoveryLedger(tmp_path).open_quarantined("cast")


def test_plan_reports_eligible_stale_without_writing(tmp_path: Path) -> None:
    from src.cast_freeze import plan_quarantine_revalidation

    source, project = make_project({"01-plain.txt": "it rained.", "02-other.txt": "it snowed."})
    try:
        prepare_cast(source, ask=bad_json_ask, max_batches=0)
        chapters = sorted((project / "chapters").glob("0[12]-*.txt"))
        ledger = RecoveryLedger(project)
        for index, path in enumerate(chapters):
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if index == 0 else "0" * 64
            ledger.record(
                "cast",
                f"legacy {index}",
                "cast_evidence_rejected",
                "legacy",
                severity="quarantine",
                evidence={"chapters": [path.name], "raw_chapter_sha256": {path.name: digest}},
            )
        before = {p.name: p.read_bytes() for p in project.iterdir() if p.is_file()}
        plan = plan_quarantine_revalidation(source, project)
        assert plan["open_quarantine_rows"] == 2
        assert plan["by_code"]["cast_evidence_rejected"] == {"eligible": 1, "stale": 1, "exhausted": 0}
        assert before == {p.name: p.read_bytes() for p in project.iterdir() if p.is_file()}
    finally:
        shutil.rmtree(project, ignore_errors=True)
        source.unlink(missing_ok=True)
