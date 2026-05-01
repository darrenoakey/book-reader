import json
import subprocess
import tempfile
from pathlib import Path

from src.arbiter_tts import tts_clone_many
from src.voice_clone import voice_path

SAMPLE_RATE = 24000


# ##################################################################
# concat wavs
# concatenate per-line wavs into a single chapter wav at SAMPLE_RATE mono
def concat_wavs(line_paths: list[Path], output_path: Path) -> None:
    if not line_paths:
        raise ValueError("No line files to concatenate")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in line_paths:
            f.write(f"file '{p}'\n")
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
# synthesize chapter
# synthesize each script line via arbiter tts-clone, then concatenate
def synthesize_chapter(script_path: Path, audio_dir: Path, voices_dir: Path) -> Path:
    chapter_name = script_path.stem
    output_path = audio_dir / f"{chapter_name}.wav"
    if output_path.exists():
        return output_path
    work_dir = audio_dir / f".lines_{chapter_name}"
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    line_paths: list[Path] = []
    with open(script_path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            speaker = list(entry.keys())[0]
            text = entry[speaker]
            ref_wav = voice_path(voices_dir, speaker)
            line_path = work_dir / f"{idx:05d}.wav"
            line_paths.append(line_path)
            if not line_path.exists():
                jobs.append({
                    "ref_wav": ref_wav,
                    "text": text,
                    "output_path": line_path,
                })
    if jobs:
        tts_clone_many(jobs)
    concat_wavs(line_paths, output_path)
    return output_path


# ##################################################################
# synthesize all chapters
# convert all scripts to audio
def synthesize_all_chapters(output_dir: Path, max_chapters: int = 0) -> list[Path]:
    script_dir = output_dir / "script"
    audio_dir = output_dir / "audio"
    voices_dir = output_dir / "voices"
    if not script_dir.exists():
        raise ValueError("script directory not found")
    if not voices_dir.exists():
        raise ValueError("voices directory not found")
    audio_dir.mkdir(parents=True, exist_ok=True)
    script_files = sorted(script_dir.glob("*.jsonl"))
    if max_chapters > 0:
        script_files = script_files[:max_chapters]
    all_created = []
    for script_path in script_files:
        output_path = synthesize_chapter(script_path, audio_dir, voices_dir)
        all_created.append(output_path)
    return all_created
