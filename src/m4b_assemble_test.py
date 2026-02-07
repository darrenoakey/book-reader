import subprocess
import tempfile
from pathlib import Path

from src.m4b_assemble import assemble_m4b, generate_silence, get_audio_duration


# ##################################################################
# create test audio
# generate a short audio file with ffmpeg
def create_test_audio(output_path: Path, duration_sec: float = 0.5) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_sec}",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg test audio failed: {result.stderr}")


# ##################################################################
# test get audio duration
# verify duration extraction works
def test_get_audio_duration() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_path = tmpdir / "test.wav"
        create_test_audio(audio_path, 1.0)
        duration = get_audio_duration(audio_path)
        assert 0.9 < duration < 1.1


# ##################################################################
# test generate silence
# verify silence generation works
def test_generate_silence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        silence_path = tmpdir / "silence.wav"
        generate_silence(0.5, silence_path)
        assert silence_path.exists()
        duration = get_audio_duration(silence_path)
        assert 0.4 < duration < 0.6


# ##################################################################
# test assemble m4b real
# creates an actual m4b file from per-chapter audio files
def test_assemble_m4b_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "test_book"
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True)
        create_test_audio(audio_dir / "00-intro.wav", 0.5)
        create_test_audio(audio_dir / "01-chapter_one.wav", 1.0)
        m4b_path = assemble_m4b(output_dir, "Test Book", "Test Author")
        assert m4b_path.exists()
        assert m4b_path.suffix == ".m4b"
        assert m4b_path.stat().st_size > 1000
        cmd = ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(m4b_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0


# ##################################################################
# test assemble m4b idempotent
# verify existing m4b is not overwritten
def test_assemble_m4b_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "test_book"
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True)
        create_test_audio(audio_dir / "00-intro.wav", 0.5)
        existing = output_dir / "test_book.m4b"
        existing.write_text("PRESERVED")
        assemble_m4b(output_dir, "Test", "Author")
        assert existing.read_text() == "PRESERVED"
