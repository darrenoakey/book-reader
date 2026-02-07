import json
import subprocess
import tempfile
from pathlib import Path

from src.audio_synth import synthesize_chapter


# ##################################################################
# test synthesize chapter real
# actually synthesizes audio for a chapter using tts multi
def test_synthesize_chapter_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        voices_dir = output_dir / "voices"
        audio_dir = output_dir / "audio"
        voices_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)
        description = "A warm male voice in his thirties. Clear and articulate with slight British accent."
        cmd = [
            str(Path.home() / "src" / "tts" / "run"),
            "export-voice",
            "test_narrator",
            description,
            "-o", str(voices_dir),
            "-q", "default",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Voice export failed: {result.stderr}")
        script_path = tmpdir / "01-test_chapter.jsonl"
        lines = [
            {"test_narrator": "Chapter One. The Beginning."},
            {"test_narrator": "It was a dark and stormy night."},
        ]
        with open(script_path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        result_path = synthesize_chapter(script_path, audio_dir, voices_dir)
        assert result_path.exists()
        assert result_path.stat().st_size > 1000
        assert result_path.name == "01-test_chapter.wav"


# ##################################################################
# test synthesize chapter idempotent
# verify existing audio is not overwritten
def test_synthesize_chapter_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()
        voices_dir = tmpdir / "voices"
        voices_dir.mkdir()
        script_path = tmpdir / "01-test.jsonl"
        script_path.write_text('{"narrator": "Hello"}\n')
        existing = audio_dir / "01-test.wav"
        existing.write_text("PRESERVED")
        result_path = synthesize_chapter(script_path, audio_dir, voices_dir)
        assert existing.read_text() == "PRESERVED"
        assert result_path == existing
