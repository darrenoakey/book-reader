import json
import subprocess
import tempfile
from pathlib import Path

TTS_RUN = Path.home() / "src" / "tts" / "run"
PAUSE_BETWEEN_SPEAKERS_MS = 500
PAUSE_BETWEEN_CHAPTERS_MS = 3000


# ##################################################################
# synthesize line
# convert a single script line to audio using tts
def synthesize_line(speaker: str, text: str, output_path: Path, voices_dir: Path) -> None:
    if output_path.exists():
        return
    voice_path = voices_dir / f"{speaker}.voice.zip"
    if not voice_path.exists():
        raise ValueError(f"Voice file not found for {speaker}: {voice_path}")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(text)
        text_file = Path(f.name)
    try:
        cmd = [
            str(TTS_RUN),
            "tts",
            str(text_file),
            "-o", str(output_path),
            "-v", str(voice_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"TTS failed: {result.stderr}")
    finally:
        text_file.unlink(missing_ok=True)


# ##################################################################
# synthesize chapter
# convert a chapter script to audio files
def synthesize_chapter(script_path: Path, audio_dir: Path, voices_dir: Path) -> list[Path]:
    chapter_name = script_path.stem
    chapter_audio_dir = audio_dir / chapter_name
    chapter_audio_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    created = []
    for i, entry in enumerate(lines):
        speaker = list(entry.keys())[0]
        text = entry[speaker]
        output_path = chapter_audio_dir / f"{i:04d}.wav"
        synthesize_line(speaker, text, output_path, voices_dir)
        created.append(output_path)
    return created


# ##################################################################
# synthesize all chapters
# convert all scripts to audio
def synthesize_all_chapters(output_dir: Path) -> list[Path]:
    script_dir = output_dir / "script"
    audio_dir = output_dir / "audio"
    voices_dir = output_dir / "voices"
    if not script_dir.exists():
        raise ValueError("script directory not found")
    if not voices_dir.exists():
        raise ValueError("voices directory not found")
    audio_dir.mkdir(parents=True, exist_ok=True)
    script_files = sorted(script_dir.glob("*.jsonl"))
    all_created = []
    for script_path in script_files:
        created = synthesize_chapter(script_path, audio_dir, voices_dir)
        all_created.extend(created)
    return all_created
