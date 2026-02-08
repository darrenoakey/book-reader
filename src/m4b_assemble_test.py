import subprocess
import tempfile
from pathlib import Path

import json

from src.m4b_assemble import assemble_m4b, generate_chime, generate_silence, get_audio_duration, number_duplicate_titles


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


# ##################################################################
# create test jpeg
# use ffmpeg to generate a small valid JPEG for testing
def create_test_jpeg(output_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=1x1:d=1",
        "-frames:v", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg jpeg creation failed: {result.stderr}")


# ##################################################################
# test number duplicate titles
# verify duplicate titles get numbered
def test_number_duplicate_titles() -> None:
    chapters = [
        {"title": "Intro", "start": 0, "end": 1},
        {"title": "Interlude", "start": 1, "end": 2},
        {"title": "Chapter One", "start": 2, "end": 3},
        {"title": "Interlude", "start": 3, "end": 4},
        {"title": "Chapter Two", "start": 4, "end": 5},
        {"title": "Interlude", "start": 5, "end": 6},
    ]
    number_duplicate_titles(chapters)
    titles = [ch["title"] for ch in chapters]
    assert titles == ["Intro", "Interlude 1", "Chapter One", "Interlude 2", "Chapter Two", "Interlude 3"]


# ##################################################################
# test number duplicate titles no duplicates
# verify unique titles are unchanged
def test_number_duplicate_titles_no_duplicates() -> None:
    chapters = [
        {"title": "Intro", "start": 0, "end": 1},
        {"title": "Chapter One", "start": 1, "end": 2},
        {"title": "Chapter Two", "start": 2, "end": 3},
    ]
    number_duplicate_titles(chapters)
    titles = [ch["title"] for ch in chapters]
    assert titles == ["Intro", "Chapter One", "Chapter Two"]


# ##################################################################
# test assemble m4b with duplicate titles
# verify duplicate chapter titles get numbered in the m4b
def test_assemble_m4b_with_duplicate_titles() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "test_book"
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True)
        create_test_audio(audio_dir / "00-interlude.wav", 0.5)
        create_test_audio(audio_dir / "01-chapter_one.wav", 0.5)
        create_test_audio(audio_dir / "02-interlude.wav", 0.5)
        m4b_path = assemble_m4b(output_dir, "Test Book", "Test Author")
        assert m4b_path.exists()
        cmd = ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(m4b_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        chapters = json.loads(result.stdout)["chapters"]
        titles = [ch["tags"]["title"] for ch in chapters]
        assert titles == ["Interlude 1", "Chapter One", "Interlude 2"]


# ##################################################################
# test assemble m4b with cover
# verify cover image is embedded in m4b
def test_assemble_m4b_with_cover() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "test_book"
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True)
        create_test_audio(audio_dir / "00-intro.wav", 0.5)
        create_test_audio(audio_dir / "01-chapter_one.wav", 0.5)
        create_test_jpeg(output_dir / "cover.jpeg")
        m4b_path = assemble_m4b(output_dir, "Test Book", "Test Author")
        assert m4b_path.exists()
        cmd = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(m4b_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        streams = json.loads(result.stdout)["streams"]
        video_streams = [s for s in streams if s["codec_type"] == "video"]
        assert len(video_streams) == 1
        assert video_streams[0]["disposition"]["attached_pic"] == 1


# ##################################################################
# test generate chime
# verify chime generation creates a valid audio file
def test_generate_chime() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        chime_path = tmpdir / "chime.wav"
        generate_chime(chime_path)
        assert chime_path.exists()
        duration = get_audio_duration(chime_path)
        assert 0.3 < duration < 1.0


# ##################################################################
# test generate chime idempotent
# verify existing chime is not overwritten
def test_generate_chime_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        chime_path = tmpdir / "chime.wav"
        chime_path.write_bytes(b"PRESERVED")
        generate_chime(chime_path)
        assert chime_path.read_bytes() == b"PRESERVED"
