import json
import subprocess
from pathlib import Path

TTS_RUN = Path.home() / "src" / "tts" / "run"


# ##################################################################
# clone voice
# create a voice zip file using tts export-voice
def clone_voice(name: str, description: str, output_dir: Path) -> Path:
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_path = voices_dir / f"{name}.voice.zip"
    if voice_path.exists():
        return voice_path
    cmd = [
        str(TTS_RUN),
        "export-voice",
        name,
        description,
        "-o", str(voices_dir),
        "-q", "hq",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Voice cloning failed for {name}: {result.stderr}")
    if not voice_path.exists():
        raise RuntimeError(f"Voice file not created: {voice_path}")
    return voice_path


# ##################################################################
# clone all voices
# create voice files for all characters
def clone_all_voices(output_dir: Path) -> list[Path]:
    voices_json_path = output_dir / "voices.json"
    if not voices_json_path.exists():
        raise ValueError("voices.json not found")
    voices = json.loads(voices_json_path.read_text(encoding="utf-8"))
    created = []
    for char_id, info in voices.items():
        voice_path = clone_voice(char_id, info["description"], output_dir)
        created.append(voice_path)
    return created
