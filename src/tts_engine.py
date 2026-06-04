"""Pluggable TTS engine selection.

Two engines, chosen by env ``BOOK_TTS_ENGINE``:

  kokoro  (default) — Kokoro-82M via arbiter ``tts-kokoro``. Hundreds of times
                      faster than qwen3-tts. Each character is mapped to a
                      kokoro voice (see kokoro_voices.py); no reference WAVs.
  qwen              — the original qwen3-tts path: design a reference WAV per
                      character (voice_clone) then clone per line.

The rest of the pipeline is engine-agnostic: it plans line jobs
({speaker, text, output_path}) and asks the engine to fill the WAVs and to
prepare per-character voices.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

ENGINE = os.environ.get("BOOK_TTS_ENGINE", "kokoro").lower()


# ##################################################################
# engine name
def engine_name() -> str:
    return ENGINE


# ##################################################################
# voices ready
# has the per-character voice preparation already been done for this engine?
def voices_ready(output_dir: Path) -> bool:
    if ENGINE == "kokoro":
        return (output_dir / "kokoro_voices.json").exists()
    return (output_dir / "voices").exists()


# ##################################################################
# speaker set
# the set of valid speaker ids for the current engine
def speaker_set(output_dir: Path) -> set[str]:
    import json
    if ENGINE == "kokoro":
        path = output_dir / "kokoro_voices.json"
    else:
        path = output_dir / "voices.json"
    if not path.exists():
        raise ValueError(f"{path.name} not found")
    return set(json.loads(path.read_text(encoding="utf-8")).keys())


# ##################################################################
# prepare voices
# create whatever per-character voice artifacts the engine needs
def prepare_voices(output_dir: Path) -> int:
    if ENGINE == "kokoro":
        from src.kokoro_voices import map_characters_to_voices
        map_characters_to_voices(output_dir)
        import json
        data = json.loads((output_dir / "kokoro_voices.json").read_text(encoding="utf-8"))
        return len(data)
    from src.voice_clone import clone_all_voices
    return len(clone_all_voices(output_dir))


# ##################################################################
# synthesize jobs
# fill every job's output_path WAV using the selected engine
def synthesize_jobs(jobs: list[dict], output_dir: Path) -> None:
    if not jobs:
        return
    if ENGINE == "kokoro":
        from src.arbiter_tts import tts_kokoro_many
        from src.kokoro_voices import load_voice_map
        vmap = load_voice_map(output_dir)
        default = vmap.get("narrator", ("af_heart", 1.0))
        kjobs = []
        for j in jobs:
            voice, speed = vmap.get(j["speaker"], default)
            kjobs.append({**j, "voice": voice, "speed": speed})
        tts_kokoro_many(kjobs)
        return
    # qwen: group by speaker so each ref WAV stays hot in the worker
    from src.arbiter_tts import tts_clone_many
    voices_dir = output_dir / "voices"
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for j in jobs:
        by_speaker[j["speaker"]].append(j)
    for speaker in sorted(by_speaker, key=lambda s: -len(by_speaker[s])):
        group = by_speaker[speaker]
        print(f"  speaker {speaker}: {len(group)} lines")
        tts_clone_many(group, ref_audio_path=voices_dir / f"{speaker}.wav")
