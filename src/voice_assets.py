import hashlib
import json
from pathlib import Path

from src.cast_plan import (
    SIDECAR_VOICES,
    artifact_matches_plan,
    load_plan,
    plan_is_stale,
)
from src.voice_acoustic import description_violations


# ##################################################################
# require current descriptions
# Legacy projects without a plan remain readable. Planned projects must never
# use a changed or incomplete profile/description cache to produce new audio.
def require_current_descriptions(project: Path) -> dict | None:
    if not (project / "cast_plan.json").is_file():
        if (project / SIDECAR_VOICES).exists() or (project / "kokoro_voices_plan.json").exists():
            raise ValueError(f"casting identity plan missing in {project}; refusing unbound voices")
        return None
    plan = load_plan(project)
    characters = json.loads((project / "characters.json").read_text(encoding="utf-8"))
    if plan_is_stale(project, characters) or not artifact_matches_plan(project, SIDECAR_VOICES, plan):
        raise ValueError(f"voice descriptions in {project} are stale; run the voices step before rendering")
    voices = json.loads((project / "voices.json").read_text(encoding="utf-8"))
    if set(voices) != set(plan["characters"]):
        raise ValueError(f"voice identities in {project} differ from the casting plan")
    for character, profile in plan["characters"].items():
        problems = description_violations(voices[character].get("description", ""), profile)
        if problems:
            raise ValueError(f"voice {character} contradicts the current casting plan: {'; '.join(problems)}")
    return plan


# ##################################################################
# reference manifest
# Reference clips are bound to their actual design instructions, not just a
# filename. Switching engines must not accidentally clone another engine's clip.
def reference_manifest(project: Path, engine: str) -> dict:
    path = project / f"{engine}_voices.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"invalid {engine} voice manifest in {project}")
    return value


# ##################################################################
# require matching reference
# Reject a changed description or edited clip without overwriting existing
# narration. The operator can archive/reimport the project to explicitly recast.
def require_matching_reference(project: Path, engine: str, character: str, description: str) -> None:
    wav = project / "voices" / f"{character}.wav"
    if not wav.is_file() or wav.stat().st_size < 100:
        return
    entry = reference_manifest(project, engine).get(character)
    if entry is None:
        if not (project / "cast_plan.json").is_file():
            return
        raise ValueError(f"unbound {engine} reference for {character}; archive the project and explicitly recast")
    if entry.get("description") != description:
        raise ValueError(f"stale {engine} reference for {character}: description changed; archive and explicitly recast")
    if entry.get("sha256") and entry["sha256"] != hashlib.sha256(wav.read_bytes()).hexdigest():
        raise ValueError(f"{engine} reference bytes changed for {character}; explicitly recast before rendering")


# ##################################################################
# remember reference
# Qwen previously skipped any existing WAV regardless of its design. Record
# each completed real reference atomically, keeping unrelated characters intact.
def remember_reference(project: Path, engine: str, character: str, description: str) -> None:
    wav = project / "voices" / f"{character}.wav"
    manifest = reference_manifest(project, engine)
    manifest[character] = {
        "ref_wav": str(wav.relative_to(project)),
        "description": description,
        "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
    }
    path = project / f"{engine}_voices.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


# ##################################################################
# require materialized voices
# Validate even when a caller would skip all completed chapters or references.
def require_materialized_voices(project: Path, engine: str, speakers: set[str] | None = None) -> None:
    plan = require_current_descriptions(project)
    if plan is None:
        return
    voices = json.loads((project / "voices.json").read_text(encoding="utf-8"))
    if engine == "kokoro":
        from src.cast_plan import SIDECAR_KOKORO

        if not artifact_matches_plan(project, SIDECAR_KOKORO, plan):
            raise ValueError(f"stale Kokoro mapping in {project}; prepare voices before rendering")
        return
    for character in speakers if speakers is not None else voices:
        wav = project / "voices" / f"{character}.wav"
        if not wav.is_file() or wav.stat().st_size < 100:
            raise ValueError(f"missing {engine} reference for {character} in {project}")
        require_matching_reference(project, engine, character, voices[character]["description"])


# ##################################################################
# bind rendered audio
# Keep completed hours immutable. Adding speakers is safe, but changing a
# speaker's actual synthesis inputs must never reuse their old cached audio.
def require_audio_binding(project: Path, directory: Path, engine: str) -> None:
    if require_current_descriptions(project) is None:
        return
    require_materialized_voices(project, engine)
    voices = json.loads((project / "voices.json").read_text(encoding="utf-8"))
    mapping = reference_manifest(project, "kokoro") if engine == "kokoro" else {}
    current = {}
    for character, voice in voices.items():
        if engine == "kokoro":
            spec = mapping[character]
            inputs = {"engine": engine, "voice": spec["voice"], "speed": spec.get("speed", 1.0)}
        else:
            wav = project / "voices" / f"{character}.wav"
            inputs = {"engine": engine, "description": voice["description"], "reference": hashlib.sha256(wav.read_bytes()).hexdigest()}
        current[character] = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    path = directory / "voices_fingerprint.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    has_audio = any(directory.rglob("*.wav"))
    if stored is None and has_audio:
        raise ValueError(f"unbound rendered audio in {directory}; archive and explicitly recast before reuse")
    if stored is not None:
        changed = [character for character, fingerprint in current.items() if character in stored and stored[character] != fingerprint]
        if changed and has_audio:
            raise ValueError(f"voice inputs changed for {changed} with rendered audio in {directory}; archive and explicitly recast")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**(stored or {}), **current}, indent=2), encoding="utf-8")
