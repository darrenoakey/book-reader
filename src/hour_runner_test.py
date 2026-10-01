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
    load_ledger,
    script_for_chapter,
    synthesize_window,
    validate_appearances,
)


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
