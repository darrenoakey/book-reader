"""Real filesystem and ffmpeg tests for bounded hourly production helpers."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.hour_runner import HOUR_MAX_SECONDS, chapter_order, load_ledger, synthesize_window


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
# test window respects cap
# choose only complete real spoken-piece WAVs that fit the remaining strict duration budget.
def test_synthesis_window_respects_remaining_duration() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        hour = root / "hour"
        lines = hour / "audio" / ".lines_00001-part"
        lines.mkdir(parents=True)
        tone_wav(lines / "00000.wav", 0.8)
        tone_wav(lines / "00001.wav", 0.8)
        (hour / "script").mkdir(parents=True)
        script = hour / "script" / "00001-part.jsonl"
        script.write_text('{"narrator": "One."}\n{"narrator": "Two."}\n', encoding="utf-8")
        (hour / "breeze_voices.json").write_text('{"narrator": {}}', encoding="utf-8")
        selected, count = synthesize_window(hour, script, 1.0, 0)
        assert count == 1
        assert len(selected) == 1
        assert selected[0]["duration"] <= HOUR_MAX_SECONDS
