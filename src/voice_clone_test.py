import tempfile
from pathlib import Path

from src.voice_clone import clone_voice


# ##################################################################
# test clone voice real
# actually creates a voice file using tts export-voice
def test_clone_voice_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True)
        description = "A warm male voice in his thirties with a slight British accent. Clear and articulate."
        voice_path = clone_voice("test_narrator", description, output_dir)
        assert voice_path.exists()
        assert voice_path.suffix == ".zip"
        assert voice_path.stat().st_size > 1000


# ##################################################################
# test clone voice idempotent
# verify existing voice file is not overwritten
def test_clone_voice_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        voices_dir = output_dir / "voices"
        voices_dir.mkdir(parents=True)
        existing = voices_dir / "existing.voice.zip"
        existing.write_text("PRESERVED")
        result = clone_voice("existing", "Any description", output_dir)
        assert result.read_text() == "PRESERVED"
