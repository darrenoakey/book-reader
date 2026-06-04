import json
import subprocess
from pathlib import Path

from arbiter_client import ArbiterClient

from src.arbiter_tts import _fetch, _submit, tts_design_to_file

# Spark host where arbiter runs and the on-disk speaker registry lives.
SPARK_HOST = "darren@10.0.0.254"
SPARK_REFS_DIR = "/home/darren/src/arbiter/local_output/refs"

REF_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Sphinx of black quartz, judge my vow. "
    "She sells seashells by the seashore on a sunny afternoon. "
    "How vexingly quick daft zebras jump."
)


# ##################################################################
# clone voice
# generate a single reference WAV per character via tts-design (called once)
def clone_voice(name: str, description: str, output_dir: Path) -> Path:
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    wav_path = voices_dir / f"{name}.wav"
    if wav_path.exists() and wav_path.stat().st_size >= 100:
        return wav_path
    tts_design_to_file(description, REF_SAMPLE_TEXT, wav_path, temperature=0.9)
    return wav_path


# ##################################################################
# register speaker
# upload one ref WAV to spark at refs/<speaker_id>.wav so the tts-clone
# adapter can find it by speaker_id alone, without any per-job ref audio
# being shipped over the wire.
def register_speaker(speaker_id: str, local_wav: Path) -> None:
    remote = f"{SPARK_HOST}:{SPARK_REFS_DIR}/{speaker_id}.wav"
    subprocess.run(["ssh", SPARK_HOST, f"mkdir -p {SPARK_REFS_DIR}"], check=True)
    subprocess.run(["scp", "-q", str(local_wav), remote], check=True)


# ##################################################################
# clone all voices
# generate reference WAVs for every character in voices.json (parallel)
# and register each one on spark by speaker_id so subsequent tts-clone
# jobs only need to send the speaker_id
def clone_all_voices(output_dir: Path) -> list[Path]:
    voices_json_path = output_dir / "voices.json"
    if not voices_json_path.exists():
        raise ValueError("voices.json not found")
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    voices = json.loads(voices_json_path.read_text(encoding="utf-8"))
    client = ArbiterClient(timeout=60)
    pending: list[dict] = []
    paths: list[tuple[str, Path]] = []
    for char_id, info in voices.items():
        wav_path = voices_dir / f"{char_id}.wav"
        paths.append((char_id, wav_path))
        if wav_path.exists() and wav_path.stat().st_size >= 100:
            continue
        params = {
            "text": REF_SAMPLE_TEXT,
            "instruct": info["description"],
            "language": "English",
            "temperature": 0.9,
            "force": True,
        }
        jid = _submit(client, "tts-design", params)
        pending.append({"job_id": jid, "output_path": wav_path, "params": params})
    for p in pending:
        _fetch(client, p["job_id"], "tts-design", p["params"], p["output_path"])
    # All ref WAVs exist locally now — register each on spark by speaker_id.
    for char_id, wav_path in paths:
        register_speaker(char_id, wav_path)
    return [wav for _, wav in paths]


# ##################################################################
# voice path
# return the local wav path for a character voice
def voice_path(voices_dir: Path, name: str) -> Path:
    p = voices_dir / f"{name}.wav"
    if not p.exists():
        raise ValueError(f"voice file not found for {name}: {p}")
    return p
