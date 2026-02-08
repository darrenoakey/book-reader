import json
import subprocess
import tempfile
from pathlib import Path

PAUSE_BETWEEN_CHAPTERS_SEC = 3.0
PAUSE_AFTER_CHIME_SEC = 0.5
PAUSE_AFTER_ANNOUNCEMENT_SEC = 1.0
SAMPLE_RATE = 24000

TTS_RUN = Path.home() / "src" / "tts" / "run"


# ##################################################################
# number duplicate titles
# append numbers to titles that appear more than once
def number_duplicate_titles(chapters: list[dict]) -> None:
    from collections import Counter
    counts = Counter(ch["title"] for ch in chapters)
    duplicates = {title for title, count in counts.items() if count > 1}
    seen: dict[str, int] = {}
    for ch in chapters:
        title = ch["title"]
        if title in duplicates:
            seen[title] = seen.get(title, 0) + 1
            ch["title"] = f"{title} {seen[title]}"


# ##################################################################
# find cover image
# look for cover image in output directory
def find_cover_image(output_dir: Path) -> Path | None:
    for name in ("cover.jpeg", "cover.jpg", "cover.png"):
        path = output_dir / name
        if path.exists():
            return path
    return None


# ##################################################################
# generate chime
# create a pleasant 3-note chime sound effect
def generate_chime(output_path: Path) -> None:
    if output_path.exists():
        return
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=523:sample_rate={SAMPLE_RATE}:duration=0.2",
        "-f", "lavfi", "-i", f"sine=frequency=659:sample_rate={SAMPLE_RATE}:duration=0.2",
        "-f", "lavfi", "-i", f"sine=frequency=784:sample_rate={SAMPLE_RATE}:duration=0.4",
        "-filter_complex",
        "[0]afade=t=in:d=0.02,afade=t=out:st=0.15:d=0.05[a];"
        "[1]afade=t=in:d=0.02,afade=t=out:st=0.15:d=0.05[b];"
        "[2]afade=t=in:d=0.02,afade=t=out:st=0.3:d=0.1[c];"
        "[a][b][c]concat=n=3:v=0:a=1",
        "-ar", str(SAMPLE_RATE),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg chime generation failed: {result.stderr}")


# ##################################################################
# generate announcement
# use tts to speak a chapter announcement with the narrator voice
def generate_announcement(text: str, narrator_voice: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({str(narrator_voice): text}, ensure_ascii=False) + "\n")
        temp_jsonl = Path(f.name)
    try:
        work_dir = output_path.parent / f".work_announce_{output_path.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(TTS_RUN), "multi",
            str(temp_jsonl),
            "-o", str(output_path),
            "-w", str(work_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"TTS announcement failed: {result.stderr}")
    finally:
        temp_jsonl.unlink(missing_ok=True)


# ##################################################################
# generate all announcements
# create chime and announcement wavs for each chapter
def generate_all_announcements(output_dir: Path, title: str, chapter_files: list[Path]) -> None:
    audio_dir = output_dir / "audio"
    voices_dir = output_dir / "voices"
    narrator_voice = voices_dir / "narrator.voice.zip"
    if not narrator_voice.exists():
        return
    chime_path = audio_dir / "chime.wav"
    generate_chime(chime_path)
    for chapter_file in chapter_files:
        announce_path = audio_dir / f"{chapter_file.stem}.announce.wav"
        if announce_path.exists():
            continue
        chapter_name = chapter_file.stem.split("-", 1)[-1].replace("_", " ").title()
        if chapter_name.lower() == "intro":
            continue
        announce_text = f"{title}. {chapter_name}."
        generate_announcement(announce_text, narrator_voice, announce_path)


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
def generate_silence(duration_sec: float, output_path: Path, sample_rate: int = SAMPLE_RATE) -> None:
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
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
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
    chapter_files = sorted(f for f in audio_dir.glob("*.wav") if not f.stem.endswith(".announce") and f.stem != "chime")
    if max_chapters > 0:
        chapter_files = chapter_files[:max_chapters]
    if not chapter_files:
        raise ValueError("No chapter audio files found")
    generate_all_announcements(output_dir, title, chapter_files)
    chime_path = audio_dir / "chime.wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        chapter_silence_path = work_dir / "chapter_silence.wav"
        generate_silence(PAUSE_BETWEEN_CHAPTERS_SEC, chapter_silence_path)
        chime_silence_path = work_dir / "chime_silence.wav"
        generate_silence(PAUSE_AFTER_CHIME_SEC, chime_silence_path)
        announce_silence_path = work_dir / "announce_silence.wav"
        generate_silence(PAUSE_AFTER_ANNOUNCEMENT_SEC, announce_silence_path)
        interleaved = []
        chapter_info = []
        current_time = 0.0
        for chapter_file in chapter_files:
            chapter_name = chapter_file.stem.split("-", 1)[-1].replace("_", " ").title()
            announce_path = audio_dir / f"{chapter_file.stem}.announce.wav"
            chapter_start = current_time
            if chime_path.exists() and announce_path.exists():
                interleaved.append(chime_path)
                current_time += get_audio_duration(chime_path)
                interleaved.append(chime_silence_path)
                current_time += PAUSE_AFTER_CHIME_SEC
                interleaved.append(announce_path)
                current_time += get_audio_duration(announce_path)
                interleaved.append(announce_silence_path)
                current_time += PAUSE_AFTER_ANNOUNCEMENT_SEC
            duration = get_audio_duration(chapter_file)
            chapter_info.append({
                "title": chapter_name,
                "start": chapter_start,
                "end": current_time + duration,
            })
            interleaved.append(chapter_file)
            current_time += duration
            interleaved.append(chapter_silence_path)
            current_time += PAUSE_BETWEEN_CHAPTERS_SEC
        number_duplicate_titles(chapter_info)
        full_audio_path = work_dir / "full.wav"
        concatenate_audio_files(interleaved, full_audio_path)
        metadata_path = work_dir / "metadata.txt"
        write_chapter_metadata(chapter_info, metadata_path)
        cover_path = find_cover_image(output_dir)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(full_audio_path),
        ]
        if cover_path:
            cmd += ["-i", str(cover_path)]
        cmd += [
            "-i", str(metadata_path),
        ]
        metadata_index = 2 if cover_path else 1
        cmd += [
            "-map_metadata", str(metadata_index),
            "-metadata", f"title={title}",
            "-metadata", f"artist={author}",
            "-metadata", f"album={title}",
            "-c:a", "aac",
            "-b:a", "64k",
        ]
        if cover_path:
            cmd += [
                "-map", "0:a",
                "-map", "1:v",
                "-c:v", "mjpeg",
                "-disposition:v:0", "attached_pic",
            ]
        cmd.append(str(m4b_path))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg m4b creation failed: {result.stderr}")
    return m4b_path
