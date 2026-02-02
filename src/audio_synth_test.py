import subprocess
import tempfile
from pathlib import Path

from src.audio_synth import synthesize_line


# ##################################################################
# test synthesize line real
# actually synthesizes audio using tts
def test_synthesize_line_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        voices_dir = output_dir / "voices"
        voices_dir.mkdir(parents=True)
        description = "A warm male voice in his thirties. Clear and articulate with slight British accent."
        cmd = [
            str(Path.home() / "src" / "tts" / "run"),
            "export-voice",
            "test_voice",
            description,
            "-o", str(voices_dir),
            "-q", "hq",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Voice export failed: {result.stderr}")
        audio_path = output_dir / "test.wav"
        synthesize_line("test_voice", "Hello world. This is a test.", audio_path, voices_dir)
        assert audio_path.exists()
        assert audio_path.stat().st_size > 1000


# ##################################################################
# test synthesize line idempotent
# verify existing audio is not overwritten
def test_synthesize_line_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        voices_dir = output_dir / "voices"
        voices_dir.mkdir(parents=True)
        existing = output_dir / "existing.wav"
        existing.write_text("PRESERVED")
        voice_file = voices_dir / "test.voice.zip"
        voice_file.write_text("dummy")
        synthesize_line("test", "text", existing, voices_dir)
        assert existing.read_text() == "PRESERVED"
