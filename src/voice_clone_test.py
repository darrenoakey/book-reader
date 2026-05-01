import tempfile
from pathlib import Path

from src.voice_clone import clone_voice, voice_path


# ##################################################################
# test clone voice real
# actually creates a voice wav by calling arbiter tts-design
def test_clone_voice_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True)
        description = "A warm male voice in his thirties with a slight British accent. Clear and articulate."
        wav_path = clone_voice("test_narrator", description, output_dir)
        assert wav_path.exists()
        assert wav_path.suffix == ".wav"
        assert wav_path.stat().st_size > 1000
        assert voice_path(output_dir / "voices", "test_narrator") == wav_path


# ##################################################################
# test clone voice idempotent
# verify existing voice files are not overwritten
def test_clone_voice_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        voices_dir = output_dir / "voices"
        voices_dir.mkdir(parents=True)
        existing_wav = voices_dir / "existing.wav"
        existing_wav.write_text("PRESERVED_WAV")
        result = clone_voice("existing", "Any description", output_dir)
        assert existing_wav.read_text() == "PRESERVED_WAV"
        assert result == existing_wav
