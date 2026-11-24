#!/usr/bin/env python3
"""dazpycheck: no-test-required

Detached narrated proof
The gate cannot run this (real model calls, minutes not seconds), so it lives
here as an explicitly-invoked proof. It renders, with the real TTS engine:

  1. isolated synthetic prose at the default narrator (deep male),
  2. a chapter announcement at the default narrator,
  3. isolated synthetic prose at an explicit override (female),

and asserts each produced a real, decodable, non-silent WAV while the
descriptions match the configured policy. Run with a bounded external timeout:

    .venv/bin/python src/narrator_voice_proof.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import narrator_voice, tts_engine

PROSE = "The harbour lights came on one by one, and the boats came home."


# ##################################################################
# wav evidence
# duration and a crude distinct-sample count prove real speech, not silence.
def wav_evidence(path: Path) -> dict:
    with wave.open(str(path)) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        payload = handle.readframes(frames)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "seconds": round(frames / rate, 3),
        "rate": rate,
        "channels": channels,
        "width": width,
        "distinct_bytes": len(set(payload)),
    }


# ##################################################################
# prepare the narrator voice
# The real voices step: design the narrator's reference clip from the policy's
# description, which is exactly what a production does before synthesis.
def prepare_narrator(project: Path, resolved: str) -> dict:
    from src.breeze_voices import prepare_breeze_voices

    project.mkdir(parents=True, exist_ok=True)
    (project / "voices.json").write_text(
        json.dumps({"narrator": {"description": narrator_voice.description(resolved)}}), encoding="utf-8"
    )
    manifest = json.loads(prepare_breeze_voices(project).read_text(encoding="utf-8"))
    entry = manifest["narrator"]
    reference = project / entry["ref_wav"]
    return {"ref_wav": str(reference), "evidence": wav_evidence(reference)}


# ##################################################################
# narrate one clip
# Render through the real engine with the narrator voice prepared for `choice`.
def narrate(text: str, choice: str, project: Path) -> dict:
    project.mkdir(parents=True, exist_ok=True)
    (project / narrator_voice.PROJECT_FILE).write_text(choice + "\n", encoding="utf-8")
    resolved = narrator_voice.resolve_for_project(project)
    prepared = prepare_narrator(project, resolved)
    out = project / f"{abs(hash(text)) % 10**8}.wav"
    tts_engine.synthesize_jobs([{"speaker": "narrator", "text": text, "output_path": out}], project)
    evidence = wav_evidence(out)
    evidence["requested"] = choice
    evidence["resolved"] = resolved
    evidence["description"] = narrator_voice.description(resolved)
    evidence["text"] = text
    evidence["reference"] = prepared
    return evidence


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="narrator-proof")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "narrator-voice-proof")
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "engine": tts_engine.engine_name(),
        "configured_default": narrator_voice.configured_default(),
        "prose_default": narrate(PROSE, narrator_voice.DEFAULT_NARRATOR_VOICE, out / "prose-default"),
        "announcement_default": narrate(
            "Synthetic Book. Chapter One.", narrator_voice.DEFAULT_NARRATOR_VOICE, out / "announcement-default"
        ),
        "prose_override": narrate(PROSE, narrator_voice.FEMALE, out / "prose-override"),
    }

    problems = []
    for key, clip in results.items():
        if not isinstance(clip, dict):
            continue
        if clip["bytes"] < 100:
            problems.append(f"{key}: clip is empty")
        if clip["seconds"] < 0.3:
            problems.append(f"{key}: clip is {clip['seconds']}s long")
        if clip["distinct_bytes"] < 16:
            problems.append(f"{key}: clip looks like digital silence")
    if results["prose_default"]["description"] != narrator_voice.description(narrator_voice.DEFAULT_NARRATOR_VOICE):
        problems.append("default prose did not use the default narrator description")
    if results["prose_override"]["resolved"] != narrator_voice.FEMALE:
        problems.append("explicit override did not resolve to the requested narrator")
    if results["prose_override"]["description"] == results["prose_default"]["description"]:
        problems.append("override produced the same narrator description as the default")

    results["problems"] = problems
    results["ok"] = not problems
    verdict = out / "verdict.json"
    verdict.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
