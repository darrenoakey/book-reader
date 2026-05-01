import json
from pathlib import Path

from src.arbiter_tts import tts_design_to_file

REF_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Sphinx of black quartz, judge my vow. "
    "How vexingly quick daft zebras jump."
)


# ##################################################################
# clone voice
# generate a voice reference wav via arbiter tts-design
def clone_voice(name: str, description: str, output_dir: Path) -> Path:
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    wav_path = voices_dir / f"{name}.wav"
    if wav_path.exists():
        return wav_path
    tts_design_to_file(description, REF_SAMPLE_TEXT, wav_path)
    return wav_path


# ##################################################################
# clone all voices
# create voice reference wavs for all characters
def clone_all_voices(output_dir: Path) -> list[Path]:
    voices_json_path = output_dir / "voices.json"
    if not voices_json_path.exists():
        raise ValueError("voices.json not found")
    voices = json.loads(voices_json_path.read_text(encoding="utf-8"))
    created = []
    for char_id, info in voices.items():
        wav_path = clone_voice(char_id, info["description"], output_dir)
        created.append(wav_path)
    return created


# ##################################################################
# voice path
# return the local wav path for a character voice
def voice_path(voices_dir: Path, name: str) -> Path:
    p = voices_dir / f"{name}.wav"
    if not p.exists():
        raise ValueError(f"voice file not found for {name}: {p}")
    return p
