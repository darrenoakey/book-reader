import json
import subprocess
import tempfile
from pathlib import Path

TTS_RUN = Path.home() / "src" / "tts" / "run"


# ##################################################################
# synthesize chapter
# convert a chapter script to audio using tts multi command
def synthesize_chapter(script_path: Path, audio_dir: Path, voices_dir: Path) -> Path:
    chapter_name = script_path.stem
    output_path = audio_dir / f"{chapter_name}.wav"
    if output_path.exists():
        return output_path
    lines = []
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                speaker = list(entry.keys())[0]
                text = entry[speaker]
                voice_path = voices_dir / f"{speaker}.voice.zip"
                if not voice_path.exists():
                    raise ValueError(f"Voice file not found for {speaker}: {voice_path}")
                lines.append({str(voice_path): text})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for entry in lines:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        temp_jsonl = Path(f.name)
    try:
        work_dir = audio_dir / f".work_{chapter_name}"
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(TTS_RUN),
            "multi",
            str(temp_jsonl),
            "-o", str(output_path),
            "-w", str(work_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            raise RuntimeError(f"TTS multi failed for {chapter_name}: {result.stderr}")
        if not output_path.exists():
            raise RuntimeError(f"Audio file not created: {output_path}")
    finally:
        temp_jsonl.unlink(missing_ok=True)
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
