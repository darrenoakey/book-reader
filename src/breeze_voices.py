"""Per-character Breeze TTS voice preparation.

Breeze TTS 2 has no fixed speaker bank: it creates a voice from a
natural-language description (voice design) and can then clone that voice
from a short reference clip (voice clone / direction). To keep a character's
timbre stable across hundreds of lines we:

  1. voice-DESIGN one short reference clip per character (instruction = the
     character's voice description from voices.json, cfg_scale 4 for strong
     instruction-following),
  2. save it as ``voices/<char_id>.wav`` with its exact transcript,
  3. record both in ``breeze_voices.json`` so the audio step can clone from
     the clip for every line the character speaks.

The reference text is fixed and deliberately emotional — two sentences with a
shift in intensity — so the clip demonstrates range the clone mode can draw
on. Every character reads the same words; only the voice differs.
"""

from __future__ import annotations

import json
from pathlib import Path

# ~15 words with a natural intensity shift; short clips clone best.
REFERENCE_TEXT = (
    "Listen to me. I have waited my whole life for this moment, "
    "and I am not giving up now."
)


# ##################################################################
# prepare breeze voices
# design one reference clip per character; returns breeze_voices.json path
def prepare_breeze_voices(output_dir: Path) -> Path:
    from src.arbiter_tts import tts_breeze_design_to_file

    voices_json = output_dir / "voices.json"
    if not voices_json.exists():
        raise ValueError("voices.json not found — run the voices_desc step first")
    descriptions = json.loads(voices_json.read_text(encoding="utf-8"))

    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "breeze_voices.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for index, (char_id, info) in enumerate(sorted(descriptions.items())):
        ref_wav = voices_dir / f"{char_id}.wav"
        description = (info or {}).get("description") or "A clear neutral voice."
        if char_id in manifest and ref_wav.exists() and ref_wav.stat().st_size >= 100:
            continue
        print(f"  breeze voice {index + 1}/{len(descriptions)}: {char_id}")
        tts_breeze_design_to_file(
            description=description,
            text=REFERENCE_TEXT,
            output_path=ref_wav,
            seed=1000 + index,
        )
        manifest[char_id] = {
            "ref_wav": str(ref_wav.relative_to(output_dir)),
            "ref_text": REFERENCE_TEXT,
            "description": description,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not manifest:
        raise ValueError("no voices prepared")
    return manifest_path


# ##################################################################
# load breeze manifest
# char_id -> {ref_wav (absolute), ref_text, description}
def load_breeze_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / "breeze_voices.json"
    if not manifest_path.exists():
        raise ValueError("breeze_voices.json not found — run the voices step first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for info in manifest.values():
        info["ref_wav"] = str(output_dir / info["ref_wav"])
    return manifest
