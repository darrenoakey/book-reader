import asyncio
import json
from pathlib import Path

from src.cast_plan import (
    SIDECAR_VOICES,
    artifact_matches_plan,
    build_cast_plan,
    load_plan,
    plan_is_stale,
    plan_missing_characters,
    profile_block,
    read_plan_sidecar,
    write_plan_sidecar,
)
from src.casting_llm import ask_cached_async
from src.voice_acoustic import description_violations, self_description


# ##################################################################
# query sonnet
# voice-description generation via boringstack qwen3.6 (kept the name for
# call-site compatibility — it is no longer Sonnet)
async def query_sonnet(prompt: str, cache_root: Path | str | None = None) -> str:
    return (await ask_cached_async(prompt, root=cache_root)).strip()


# ##################################################################
# parse json response
# extract json from claude response handling markdown code blocks
def parse_json_response(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            block = text[start : end + 3]
            lines = block.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
    if not text.startswith("{"):
        brace_pos = text.find("{")
        if brace_pos != -1:
            text = text[brace_pos:]
            end_brace = text.rfind("}")
            if end_brace != -1:
                text = text[: end_brace + 1]
    return json.loads(text)


# ##################################################################
# ensure cast plan
# a fresh plan is built for a new project; a stale fingerprint (changed
# characters/source context) regenerates it BEFORE any description call so no
# description is ever produced against — or silently reused from — an old plan.
async def ensure_cast_plan(output_dir: Path, characters: dict, cache_root: Path | str | None = None) -> dict:
    missing = plan_missing_characters(output_dir, characters)
    if plan_is_stale(output_dir, characters) or missing:
        return await build_cast_plan(output_dir, characters, cache_root=cache_root)
    return load_plan(output_dir)


# ##################################################################
# plan constraints block
# the persisted whole-cast plan row for this character: gender, age band,
# accent (cohort-coherent), vocal size and distinctness are constraints the
# description MUST satisfy, so parallel calls cannot drift apart.
def _constraint_text(plan: dict, char_id: str) -> str:
    block = profile_block(plan, char_id)
    if not block:
        return "No cast-plan profile exists for this character; do not invent demographic details."
    return block


# ##################################################################
# bounded deterministic description repair
# the SAME boundary every engine's materialized descriptors pass through:
# known gender/age/accent from the plan must survive the description text,
# with ONE bounded model re-ask; a still-contradicting description fails
# closed (better no clip than a wrong-gender one).
async def _validated_description(char_id: str, prompt: str, plan: dict | None, cache_root: Path | str | None) -> str:
    text = self_description(await query_sonnet(prompt, cache_root))
    if plan is None:
        return text
    profile = plan.get("characters", {}).get(char_id)
    if profile is None:
        return text
    violations = description_violations(text, profile)
    if not violations:
        return text
    print(f"  {char_id}: description contradicts cast plan ({'; '.join(violations)}) — one bounded repair")
    repair = (
        prompt + "\n\nYour previous reply was:\n" + text
        + "\n\nIt contradicted the casting plan:\n"
        + "\n".join(f"- {v}" for v in violations)
        + "\n\nRewrite the voice description so it satisfies EVERY constraint exactly "
        "(state the correct gender, age band and accent explicitly, including a literal "
        f"age cue for {profile.get('age_band', 'unknown')!r}). Output the voice "
        "description text only."
    )
    text2 = self_description(await query_sonnet(repair, cache_root))
    if not description_violations(text2, profile):
        return text2
    raise ValueError(
        f"voice description for {char_id} contradicts the cast plan after repair: "
        + "; ".join(description_violations(text2, profile))
    )


async def _voice_description_for_one(
    char_id: str, info: dict, plan: dict | None = None, cache_root: Path | str | None = None
) -> tuple[str, dict]:
    bio = info.get("bio") or info.get("description") or ""
    is_narrator = char_id == "narrator"
    role_hint = (
        (
            "Audiobook narrator. Focus on narration qualities: clarity, authority, "
            "warmth, pace, range for character voices."
        )
        if is_narrator
        else "Audiobook character voice for TTS cloning."
    )
    constraints = ""
    if plan is not None:
        constraints = f"""
CAST PLAN CONSTRAINTS (authoritative — the description MUST match all of these;
use "unknown" fields as freedom, never invent a value for them):
{_constraint_text(plan, char_id)}
"""
    prompt = f"""You are designing a voice for an audiobook character. Output a concise (60-100 word) voice description suitable for TTS voice cloning.

{role_hint}
{constraints}
Character id: {char_id}
Bio: {bio}

Cover: gender, age, accent/dialect, pitch, timbre, pace, emotion, distinctive vocal traits. Be specific and vivid. State the character's gender, age and accent explicitly when the cast plan gives them. Never derive pitch, accent or timbre from race or ethnicity.

Output the voice description text only. No JSON, no preamble."""
    text = await _validated_description(char_id, prompt, plan, cache_root)
    if not text:
        raise ValueError(f"empty voice description for {char_id}")
    return char_id, {"description": text}


# ##################################################################
# generate all voice descriptions
# per-character parallel calls (batched JSON triggers refusal), each one
# constrained by the same persisted whole-cast plan
async def generate_all_voice_descriptions(
    characters: dict, plan: dict | None = None, cache_root: Path | str | None = None
) -> dict:
    print(f"  voice descriptions: {len(characters)} parallel sonnet calls")
    results = await asyncio.gather(
        *(_voice_description_for_one(cid, characters[cid], plan, cache_root) for cid in sorted(characters))
    )
    return {cid: desc for cid, desc in results}


# ##################################################################
# voices freshness
# voices.json is fresh while its fingerprint sidecar matches the current plan
def _voices_fresh(output_dir: Path, plan_fingerprint: dict) -> bool:
    sidecar = read_plan_sidecar(output_dir, SIDECAR_VOICES)
    if sidecar is None or sidecar.get("fingerprint") != plan_fingerprint:
        return False
    voices_path = output_dir / "voices.json"
    if not voices_path.is_file():
        return False
    try:
        voices = json.loads(voices_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    plan = load_plan(output_dir)
    return (
        isinstance(voices, dict) and set(voices) == set(plan["characters"])
        and all(isinstance(v, dict) and isinstance(v.get("description"), str) and v["description"].strip() for v in voices.values())
        and artifact_matches_plan(output_dir, SIDECAR_VOICES, plan)
    )


# ##################################################################
# generate voices
# main entry point to create voices.json from characters.json; a fresh
# (plan-matching) voices.json is reused, a stale one is regenerated — never
# silently reused. voices.json itself stays a plain per-character map (no
# metadata keys): the binding lives in the sidecar so no consumer breaks.
async def generate_voices(output_dir: Path, cache_root: Path | str | None = None) -> Path:
    characters_path = output_dir / "characters.json"
    voices_path = output_dir / "voices.json"
    if not characters_path.exists():
        raise ValueError("characters.json not found")
    characters = json.loads(characters_path.read_text(encoding="utf-8"))
    plan = await ensure_cast_plan(output_dir, characters, cache_root)
    if _voices_fresh(output_dir, plan["fingerprint"]):
        return voices_path
    if voices_path.exists():
        if any((output_dir / "voices").glob("*.wav")) or any((output_dir / "audio").glob("*.wav")):
            raise ValueError(f"casting changed in {output_dir} with materialized voices/audio; archive or reimport the project to explicitly recast without mixing old and new voices")
        print("  cast plan changed — regenerating voice descriptions (was stale)")
    from src.series_voices import inherited_descriptions

    inherited = inherited_descriptions(output_dir, characters, plan)
    fresh = {cid: info for cid, info in characters.items() if cid not in inherited}
    voices = {**await generate_all_voice_descriptions(fresh, plan, cache_root), **inherited}
    voices = {cid: voices[cid] for cid in sorted(voices)}
    voices_path.write_text(json.dumps(voices, indent=2), encoding="utf-8")
    write_plan_sidecar(output_dir, plan["fingerprint"], SIDECAR_VOICES)
    return voices_path


# ##################################################################
# generate voices sync
# synchronous wrapper for generate_voices
def generate_voices_sync(output_dir: Path, cache_root: Path | str | None = None) -> Path:
    return asyncio.run(generate_voices(output_dir, cache_root))
