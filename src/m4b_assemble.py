import subprocess
import tempfile
from pathlib import Path

PAUSE_BETWEEN_CHAPTERS_SEC = 3.0


# ##################################################################
# get audio duration
# get duration of audio file in seconds using ffprobe
def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


# ##################################################################
# generate silence
# create a silent audio file of specified duration
def generate_silence(duration_sec: float, output_path: Path, sample_rate: int = 22050) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", str(duration_sec),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg silence generation failed: {result.stderr}")


# ##################################################################
# concatenate audio files
# combine multiple audio files into one
def concatenate_audio_files(audio_files: list[Path], output_path: Path) -> None:
    if not audio_files:
        raise ValueError("No audio files to concatenate")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for audio_file in audio_files:
            f.write(f"file '{audio_file}'\n")
        list_file = Path(f.name)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
    finally:
        list_file.unlink(missing_ok=True)


# ##################################################################
# write chapter metadata
# create ffmpeg chapter metadata file
def write_chapter_metadata(chapters: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for chapter in chapters:
            f.write("\n[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={int(chapter['start'] * 1000)}\n")
            f.write(f"END={int(chapter['end'] * 1000)}\n")
            title = chapter["title"].replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\\", "\\\\")
            f.write(f"title={title}\n")


# ##################################################################
# assemble m4b
# create final m4b audiobook with chapter markers
def assemble_m4b(output_dir: Path, title: str, author: str, max_chapters: int = 0) -> Path:
    audio_dir = output_dir / "audio"
    if not audio_dir.exists():
        raise ValueError("audio directory not found")
    book_name = output_dir.name
    m4b_path = output_dir / f"{book_name}.m4b"
    if m4b_path.exists():
        return m4b_path
    chapter_files = sorted(audio_dir.glob("*.wav"))
    if max_chapters > 0:
        chapter_files = chapter_files[:max_chapters]
    if not chapter_files:
        raise ValueError("No chapter audio files found")
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        chapter_silence_path = work_dir / "chapter_silence.wav"
        generate_silence(PAUSE_BETWEEN_CHAPTERS_SEC, chapter_silence_path)
        interleaved = []
        chapter_info = []
        current_time = 0.0
        for chapter_file in chapter_files:
            duration = get_audio_duration(chapter_file)
            chapter_name = chapter_file.stem.split("-", 1)[-1].replace("_", " ").title()
            chapter_info.append({
                "title": chapter_name,
                "start": current_time,
                "end": current_time + duration,
            })
            interleaved.append(chapter_file)
            current_time += duration
            interleaved.append(chapter_silence_path)
            current_time += PAUSE_BETWEEN_CHAPTERS_SEC
        full_audio_path = work_dir / "full.wav"
        concatenate_audio_files(interleaved, full_audio_path)
        metadata_path = work_dir / "metadata.txt"
        write_chapter_metadata(chapter_info, metadata_path)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(full_audio_path),
            "-i", str(metadata_path),
            "-map_metadata", "1",
            "-metadata", f"title={title}",
            "-metadata", f"artist={author}",
            "-metadata", f"album={title}",
            "-c:a", "aac",
            "-b:a", "64k",
            str(m4b_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg m4b creation failed: {result.stderr}")
    return m4b_path
