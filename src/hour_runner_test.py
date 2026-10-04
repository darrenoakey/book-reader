"""Real filesystem and ffmpeg tests for bounded hourly production helpers."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.hour_runner import (
    HOUR_MAX_SECONDS,
    appearance_schema,
    canonical_character_id,
    chapter_order,
    extend_appearances,
    load_lean_cast,
    load_ledger,
    prepare_hour_directory,
    script_for_chapter,
    synthesize_window,
    validate_appearances,
)


def test_real_lean_hour_cast_loads_existing_212_actor_summary() -> None:
    root = Path("/Users/darrenoakey/src/book-reader")
    source = root / "incoming" / "weakest_beast_tamer.txt"
    project = root / "output" / "weakest_beast_tamer"
    cast, aliases = load_lean_cast(source, project)
    assert len(cast) == 212 and len(aliases) >= len(cast)
    assert all({"name", "bio", "look"} <= set(info) for info in cast.values())


# ##################################################################
# tone wav
# create genuine 24 kHz audio for the selection logic instead of an invented media artifact.
def tone_wav(path: Path, seconds: float) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# ##################################################################
# test numeric ordering
# prove production chapters remain correctly ordered after the two-digit boundary.
def test_numeric_chapter_order() -> None:
    paths = [Path("00099-part.txt"), Path("00100-part.txt"), Path("00009-part.txt")]
    assert [path.name for path in sorted(paths, key=chapter_order)] == [
        "00009-part.txt",
        "00099-part.txt",
        "00100-part.txt",
    ]


# ##################################################################
# test display-name aliases
# preserve an established identity for an equivalent display name while keeping distinct near-spellings separate.
def test_display_name_aliases_do_not_merge_near_names() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cast = {"ren": {"name": "Ren"}, "ron": {"name": "Ron"}, "captain": {"name": "Captain Vale"}}
        assert canonical_character_id(root, cast, "captain_vale", "CAPTAIN VALE") == "captain"
        assert canonical_character_id(root, cast, "ren_alias", "Ren") == "ren"
        assert canonical_character_id(root, cast, "ron_alias", "Ron") == "ron"


# ##################################################################
# test appearance schema validation
# enforce exact requested identities and reject partial or fabricated cache additions before any state write.
def test_appearance_schema_requires_exact_nonempty_identity_set() -> None:
    schema = appearance_schema(["ren", "father"])
    assert schema["required"] == ["ren", "father"]
    assert schema["additionalProperties"] is False
    assert validate_appearances({"ren": "dark hair", "father": "gray beard"}, ["ren", "father"]) == {
        "ren": "dark hair",
        "father": "gray beard",
    }
    with pytest.raises(ValueError):
        validate_appearances({"ren": "dark hair"}, ["ren", "father"])
    with pytest.raises(ValueError):
        validate_appearances({"ren": "dark hair", "father": "", "extra": "invented"}, ["ren", "father"])


# ##################################################################
# test existing appearance identity remains byte stable
# a fully-cached project must not call the model or rewrite established appearance identities.
def test_extend_appearances_keeps_cached_identity_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        path = project / "appearances.json"
        original = b'{"ren":"established dark hair"}\n'
        path.write_bytes(original)
        extend_appearances(project, {"narrator": {"name": "Narrator"}, "ren": {"look": "ignored"}})
        assert path.read_bytes() == original


# ##################################################################
# test native appearance generation real
# use a namespaced cast fixture against native Ollama and prove only the missing identity is atomically appended.
def test_extend_appearances_real_native() -> None:
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        original = {"ren": "established black hair and plain traveling clothes"}
        (project / "appearances.json").write_text(json.dumps(original), encoding="utf-8")
        cast = {
            "narrator": {"name": "Narrator"},
            "ren": {"name": "Ren", "look": "must remain established"},
            "father": {
                "name": "Father",
                "look": "An older ordinary human man with weathered face, gray hair, and work clothes.",
            },
        }
        extend_appearances(project, cast)
        appearances = json.loads((project / "appearances.json").read_text(encoding="utf-8"))
        assert appearances["ren"] == original["ren"]
        assert isinstance(appearances["father"], str) and appearances["father"].strip()


# ##################################################################
# test frozen hour metadata view
# keeps legacy aliases in the root cache but exposes only approved canonical IDs to all Hour 3 storyboard and TTS readers.
def test_prepare_frozen_hour_excludes_inactive_legacy_metadata() -> None:
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        (project / "chapters").mkdir()
        (project / "chapters" / "00-intro.txt").write_text("Book by Author", encoding="utf-8")
        for name, value in (
            ("characters.json", {"tiger_boy": {"name": "Tiger Boy"}, "gene": {"name": "Gene"}}),
            ("voices.json", {"tiger_boy": {"description": "canonical"}, "gene": {"description": "legacy"}}),
            (
                "breeze_voices.json",
                {"tiger_boy": {"ref_wav": "voices/tiger_boy.wav"}, "gene": {"ref_wav": "voices/gene.wav"}},
            ),
            ("appearances.json", {"tiger_boy": "canonical appearance", "gene": "legacy appearance"}),
        ):
            (project / name).write_text(json.dumps(value), encoding="utf-8")
        hour = project / "hours/hour-003"
        prepare_hour_directory(project, hour, {"tiger_boy": {"name": "Tiger Boy"}, "narrator": {"name": "Narrator"}})
        assert set(json.loads((hour / "characters.json").read_text(encoding="utf-8"))) == {"tiger_boy"}
        assert set(json.loads((hour / "voices.json").read_text(encoding="utf-8"))) == {"tiger_boy"}
        assert set(json.loads((hour / "appearances.json").read_text(encoding="utf-8"))) == {"tiger_boy"}
        assert (hour / "frozen_cast_active_only.txt").read_text(encoding="utf-8") == "true\n"


# ##################################################################
# test frozen alias script copy
# keeps the historical immutable script byte-for-byte while producing a source-exact frozen copy that sends only the canonical original voice to new hours.
def test_script_cache_resolves_frozen_alias_without_rewriting_history() -> None:
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        chapter = project / "chapters" / "01-part.txt"
        chapter.parent.mkdir(parents=True)
        source = "Gene said hello."
        chapter.write_text(source, encoding="utf-8")
        cache = project / "script_cache"
        cache.mkdir()
        legacy = cache / "01-part.jsonl"
        legacy.write_text('{"gene": "Gene said hello."}\n', encoding="utf-8")
        payload = legacy.read_bytes()
        (cache / "01-part.jsonl.hour.meta.json").write_text(
            json.dumps(
                {
                    "mode": "immutable-spans",
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        original = legacy.read_bytes()
        frozen = script_for_chapter(
            project,
            project / "hours/hour-003",
            chapter,
            {"tiger_boy": {"name": "Tiger Boy"}, "narrator": {"name": "Narrator"}},
            {"gene": "tiger_boy"},
        )
        assert legacy.read_bytes() == original
        assert frozen.parent.name == "frozen_script_cache"
        assert frozen.read_text(encoding="utf-8") == '{"tiger_boy": "Gene said hello."}\n'


# ##################################################################
# test ledger source mismatch
# prevent a resumable project from silently continuing a different source file.
def test_ledger_rejects_changed_source() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "book.txt"
        source.write_text("first source", encoding="utf-8")
        ledger = load_ledger(root / "output", source)
        (root / "output").mkdir()
        (root / "output" / "hours.json").write_text(json.dumps(ledger), encoding="utf-8")
        source.write_text("different source", encoding="utf-8")
        with pytest.raises(RuntimeError, match="SHA-256"):
            load_ledger(root / "output", source)


# ##################################################################
# test ambient cli enters venv
# invoke the public facade from the ambient interpreter and prove its hour runtime import happens in the installed venv.
def test_hour_verify_only_cli_uses_venv() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "book.txt"
        source.write_text("A brief source for runtime verification.", encoding="utf-8")
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [str(project / "run"), "hour", str(source), "--verify-only"],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "sha256" in result.stdout
        assert str(source) in result.stdout


# ##################################################################
# test window respects cap
# choose only complete real spoken-piece WAVs that fit the remaining strict duration budget.
def test_synthesis_window_respects_remaining_duration() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hour = root / "hour"
        lines = root / "audio_cache" / ".lines_00001-part"
        lines.mkdir(parents=True)
        tone_wav(lines / "00000.wav", 0.8)
        tone_wav(lines / "00001.wav", 0.8)
        (hour / "script").mkdir(parents=True)
        script = hour / "script" / "00001-part.jsonl"
        script.write_text('{"narrator": "One."}\n{"narrator": "Two."}\n', encoding="utf-8")
        (root / "breeze_voices.json").write_text(
            '{"narrator": {"ref_wav": "voices/narrator.wav", "ref_text": "Listen."}}', encoding="utf-8"
        )
        selected, count = synthesize_window(root, script, 1.0, 0)
        assert count == 1
        assert len(selected) == 1
        assert selected[0]["duration"] <= HOUR_MAX_SECONDS


# ##################################################################
# test native fault then next valid chapter persists across restart
# a real corrupt canonical script quarantines only its chapter; the next real valid chapter is accepted, and a fresh ledger/process view still holds the fault and the cursor.
def test_native_fault_then_next_valid_chapter_persists_across_restart() -> None:
    from src.data_recovery import DataIssue, RecoveryLedger, is_data_error

    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        chapters = project / "chapters"
        cache = project / "script_cache"
        chapters.mkdir()
        cache.mkdir()
        texts = {"01-bad.txt": "Gene said hello.", "02-good.txt": "Gene waved."}
        for name, text in texts.items():
            (chapters / name).write_text(text, encoding="utf-8")
        for name, body in (("01-bad.jsonl", "not json at all\n"), ("02-good.jsonl", '{"gene": "Gene waved."}\n')):
            payload = body.encode()
            (cache / name).write_bytes(payload)
            source = texts[name.replace(".jsonl", ".txt")]
            (cache / f"{name}.hour.meta.json").write_text(
                json.dumps(
                    {
                        "mode": "immutable-spans",
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
        cast = {"gene": {"name": "Gene"}, "narrator": {"name": "Narrator"}}
        recovery = RecoveryLedger(project)
        accepted: list[str] = []
        cursor = 0
        for index, name in enumerate(sorted(texts)):
            try:
                script_for_chapter(project, project / "hours/hour-003", chapters / name, cast, {})
                accepted.append(name)
            except DataIssue as error:
                assert is_data_error(error)
                checkpoint = {"next_chapter": index + 1, "next_piece": 0}
                recovery.record_error("hour", name, error, checkpoint=checkpoint)
            cursor = index + 1
        assert accepted == ["02-good.txt"] and cursor == 2
        restarted = RecoveryLedger(project).entries()
        assert [row["item"] for row in restarted] == ["01-bad.txt"]
        assert restarted[0]["code"] == "canonical_script_invalid"
        assert restarted[0]["checkpoint"] == {"next_chapter": 1, "next_piece": 0}


# ##################################################################
# deferred-image hour tests (real files, namespaced output project)
def _deferred_book():
    import shutil
    import uuid

    from src.epub_extract import get_output_dir

    source = Path(tempfile.mkdtemp()) / f"deferred_test_{uuid.uuid4().hex[:10]}.txt"
    source.write_text("Chapter 1\n\nIt began.\n\nChapter 2\n\nIt went on.\n", encoding="utf-8")
    project = get_output_dir(source)
    project.mkdir(parents=True, exist_ok=True)
    return source, project, lambda: shutil.rmtree(project, ignore_errors=True)


def test_modes_exclusive_and_render_needs_saved_audio() -> None:
    from src.hour_runner import run_hour

    source, project, cleanup = _deferred_book()
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            run_hour(source, 3, defer_images=True, render_images=True)
        with pytest.raises(RuntimeError, match="no saved audio"):
            run_hour(source, 3, render_images=True)
        assert not (project / ".pipeline.lock").exists()
    finally:
        cleanup()


def test_next_hour_requires_previous_audio_complete() -> None:
    from src.hour_runner import run_hour

    source, project, cleanup = _deferred_book()
    try:
        atomic = project / "hours.json"
        atomic.write_text(
            json.dumps({"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "hours": {"2": {"complete": False}}}),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="previous hour audio is not complete"):
            run_hour(source, 3, defer_images=True)
    finally:
        cleanup()


def test_saved_audio_tamper_refused_and_text_located() -> None:
    from src.hour_runner import file_digest, locate_text, render_saved_hour

    _source, project, cleanup = _deferred_book()
    try:
        audio = project / "hours" / "hour-003" / "audio"
        audio.mkdir(parents=True)
        tone_wav(audio / "hour-00000.wav", 1)
        timeline = audio / "hour-00000.timeline.json"
        timeline.write_text(json.dumps({"lines": [{"index": 0, "speaker": "x", "text": "I lost the bet. It's official!", "start": 0.0, "end": 1.0}]}))
        entry = {"audio_complete": True, "audio_sha256": file_digest(audio / "hour-00000.wav"), "timeline_sha256": "0" * 64}
        with pytest.raises(RuntimeError, match="refusing to retime"):
            render_saved_hour(project, {"hours": {"3": entry}}, "3", "t")
        hits = locate_text(project, "i lost the bet.  it's OFFICIAL")
        assert [h["hour"] for h in hits] == ["hour-003"] and hits[0]["start"] == 0.0
    finally:
        cleanup()


def test_prepare_keeps_earlier_lean_actor_profiles_in_same_hour() -> None:
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        (project / "chapters").mkdir()
        for name, value in (
            ("characters.json", {"tiger_boy": {"name": "Tiger Boy"}}),
            ("voices.json", {}),
            ("breeze_voices.json", {}),
            ("appearances.json", {}),
        ):
            (project / name).write_text(json.dumps(value), encoding="utf-8")
        hour = project / "hours/hour-003"
        cast = {"tiger_boy": {}, "new_one": {}, "other": {}}
        prepare_hour_directory(project, hour, cast)
        local = json.loads((hour / "characters.json").read_text(encoding="utf-8"))
        local["new_one"] = {"name": "New One", "bio": "b", "look": "l"}
        (hour / "characters.json").write_text(json.dumps(local), encoding="utf-8")
        prepare_hour_directory(project, hour, cast)
        merged = json.loads((hour / "characters.json").read_text(encoding="utf-8"))
        assert set(merged) == {"tiger_boy", "new_one"} and merged["new_one"]["name"] == "New One"


def test_scene_portraits_select_only_storyboard_characters() -> None:
    from src.hour_runner import ensure_scene_portraits

    with tempfile.TemporaryDirectory() as directory:
        hour = Path(directory)
        (hour / "refs").mkdir()
        (hour / "refs" / "shown.png").write_bytes(b"png")
        (hour / "storyboard.json").write_text(
            json.dumps({"appearances": {"shown": "a look", "unused": "another look"},
                        "scenes": [{"characters": ["shown", "narrator"]}, {"characters": []}]}),
            encoding="utf-8",
        )
        assert ensure_scene_portraits(hour) == []
        assert not (hour / "refs" / "unused.png").exists()


def test_locate_text_spans_adjacent_lines_and_ellipsis() -> None:
    from src.hour_runner import locate_text

    _source, project, cleanup = _deferred_book()
    try:
        audio = project / "hours" / "hour-004" / "audio"
        audio.mkdir(parents=True)
        lines = [
            {"index": 0, "speaker": "x", "text": "Earlier words. I lost the", "start": 0.0, "end": 1.0},
            {"index": 1, "speaker": "x", "text": "bet.  Then a pause", "start": 1.0, "end": 2.0},
            {"index": 2, "speaker": "x", "text": "It\u2019s official, a new beetle.", "start": 2.0, "end": 3.0},
        ]
        (audio / "hour-00000.timeline.json").write_text(json.dumps({"lines": lines}), encoding="utf-8")
        hits = locate_text(project, "I lost the bet ... It's official")
        assert [(h["hour"], h["start"]) for h in hits] == [("hour-004", 0.0)]
        assert locate_text(project, "official lost") == []
    finally:
        cleanup()
