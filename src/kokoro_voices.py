"""Kokoro voice catalog + LLM-driven character→voice mapping.

Kokoro has no voice cloning — it ships a fixed bank of named voice packs. We
therefore *map* each character's voice description onto the closest kokoro
voice (optionally a weighted blend of two) plus a speech-rate, using the
boringstack LLM. The result is saved to ``kokoro_voices.json``:

    {"<char_id>": {"voice": "af_heart" | "af_heart*0.6+am_michael*0.4",
                    "speed": 1.0, "description": "<original description>"}}

A "voice" string is exactly what the arbiter ``tts-kokoro`` job accepts.

Casting integrity: when a whole-cast plan (src/cast_plan.py) exists, the
mapping prompt carries each profile (gender/age/accent/vocal size) and cohort,
and the result is then deterministically validated against the fixed bank —
bank voices have a gender and accent fully determined by their id prefix, so
mismatches are hard failures with bounded deterministic repair, never silently
accepted. Blends must never mix genders (or accents) across their components.
Distinctness inside a shared accent uses the measured f0 table
(src/kokoro_f0_table.py) where available.

Future (documented in arbiter/KOKORO_TTS.md): pre-evolve hundreds of custom
voices with kvoicewalk to grow the bank from 54 to ~500, then map onto that.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from src.casting_llm import ask_cached as ask_sync
from src.kokoro_f0_table import F0_MEDIANS

# Full kokoro voice bank (54). Language/gender/accent is fully determined by the
# id prefix: a=American EN, b=British EN, e=Spanish, f=French, h=Hindi,
# i=Italian, j=Japanese, p=Br-Portuguese, z=Mandarin; f=female, m=male.
ALL_VOICES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]

_LANG = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

# Default English pool we ask the LLM to map onto (most books are English).
ENGLISH_VOICES = [v for v in ALL_VOICES if v[0] in ("a", "b")]
DEFAULT_VOICE = "af_heart"  # kokoro's reference / highest-quality voice
DEFAULT_NARRATOR = "bm_george"  # warm British male default for narration

# Plan accent phrases → kokoro id prefix. Matched case-insensitively as
# substrings of the plan accent; unresolvable accents keep the American pool.
ACCENT_PREFIXES: list[tuple[str, str]] = [
    ("british", "b"),
    ("uk", "b"),
    ("english english", "b"),
    ("london", "b"),
    ("scott", "b"),
    ("welsh", "b"),
    ("irish", "b"),
    ("american", "a"),
    ("us", "a"),
    ("midwest", "a"),
    ("california", "a"),
    ("new york", "a"),
    ("southern u", "a"),
    ("spanish", "e"),
    ("french", "f"),
    ("hindi", "h"),
    ("indian", "h"),
    ("italian", "i"),
    ("japanese", "j"),
    ("brazil", "p"),
    ("portuguese", "p"),
    ("mandarin", "z"),
    ("chinese", "z"),
]


# ##################################################################
# describe voice
# human-readable gender/accent for a voice id (certain from the prefix)
def describe_voice(voice_id: str) -> str:
    lang = _LANG.get(voice_id[0], "English")
    gender = "female" if voice_id[1] == "f" else "male"
    return f"{lang} {gender}"


# ##################################################################
# voice parts
# ordered (id, weight) components of a voice spec
def voice_parts(spec: str) -> list[tuple[str, float]]:
    parts: list[tuple[str, float]] = []
    for part in (spec or "").split("+"):
        name, _, weight = part.partition("*")
        name = name.strip()
        if not name:
            continue
        try:
            w = float(weight) if weight.strip() else 1.0
        except ValueError:
            return []
        if not math.isfinite(w) or w <= 0:
            return []
        parts.append((name, w))
    return parts


# ##################################################################
# accent prefix for a plan accent phrase
def accent_prefix(accent: str) -> str | None:
    text = (accent or "").lower()
    if not text or text == "unknown":
        return None
    for needle, prefix in ACCENT_PREFIXES:
        if re.search(rf"\b{re.escape(needle)}\b", text):
            return prefix
    return None


# ##################################################################
# voice catalog text
# the menu we hand the LLM
def _catalog_text(voices: list[str]) -> str:
    return "\n".join(f"- {v}: {describe_voice(v)}" for v in voices)


# ##################################################################
# build prompt
# ask the LLM to assign every character a kokoro voice + speed in one call so
# voices stay distinct across the cast; with a cast plan each row carries the
# profile + cohort constraints
def _profile_line(profile: dict | None) -> str:
    if not profile:
        return "(no cast plan profile)"
    bits = [f"gender={profile['gender']}", f"age={profile['age_band']}"]
    if profile.get("approx_age"):
        bits.append(f"~{profile['approx_age']}")
    bits.append(f"accent={profile['accent']}")
    bits.append(f"vocal_size={profile['vocal_size']}")
    return " ".join(bits)


def _build_prompt(characters: dict, voices: list[str], plan: dict | None = None) -> str:
    cast = []
    cohorts: dict[str, str] = {}
    if plan:
        for cohort in plan.get("cohorts", []):
            for member in cohort["members"]:
                cohorts[member] = f"{cohort['kind']} cohort '{cohort['id']}' — shared accent {cohort['accent']}"
    for cid, info in characters.items():
        if str(cid).startswith("_"):
            continue
        desc = info.get("description") or info.get("bio") or ""
        profile = (plan or {}).get("characters", {}).get(cid)
        line = f"- {cid}: {desc}"
        if profile:
            line += f" [PLAN: {_profile_line(profile)}]"
        if cid in cohorts:
            line += f" [COHORT: {cohorts[cid]}]"
        cast.append(line)
    cast_text = "\n".join(cast)
    catalog = _catalog_text(voices)
    plan_rules = ""
    if plan:
        plan_rules = """
- PLAN constraints are AUTHORITATIVE: match the plan gender exactly, and the
  plan accent exactly (cohort members MUST use voices with the same accent as
  each other). Vocal size and age guide pitch choice (rich/large and older →
  deeper voices in the pool; light and younger → higher voices).
- Never map pitch, accent or timbre from race or ethnicity."""
    return f"""You assign each audiobook character a voice from the fixed Kokoro voice bank.

AVAILABLE KOKORO VOICES (id: accent + gender):
{catalog}

CHARACTERS (id: voice description [+ casting plan profile]):
{cast_text}

For EACH character choose:
- "voice": a voice id from the bank that best matches the character's gender and accent.
  You MAY blend two voices for a more distinctive timbre using the format
  "id1*0.6+id2*0.4" (weights are any positive numbers; same gender recommended).
- "speed": a speech-rate multiplier between 0.85 and 1.15 (older/graver = slower,
  younger/energetic = faster; 1.0 = normal).

Rules:
- Match gender exactly. Match accent when the description implies one (British,
  American, etc.); otherwise prefer American English voices.
- A blend MUST use voices of the SAME gender and the SAME accent prefix.
- Make voices DISTINCT: do not give two different characters the same id+speed.
  Use different ids, blends, and speeds to separate same-gender characters —
  especially characters sharing one accent (separate them by pitch/timbre, not
  by leaving the accent).{plan_rules}
- "narrator" should be a clear, pleasant, authoritative voice.
- Every character id MUST appear exactly once in the output.

Output ONLY valid JSON, no markdown, no commentary, exactly this shape:
{{"<char_id>": {{"voice": "<id or blend>", "speed": <number>}}}}"""


# ##################################################################
# parse mapping
# tolerant JSON extraction
def _parse_mapping(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        a = text.find("```")
        b = text.rfind("```")
        if a != b:
            block = text[a + 3 : b].split("\n", 1)
            text = block[1] if len(block) > 1 else block[0]
    if not text.startswith("{"):
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1:
            text = text[i : j + 1]
    return json.loads(text)


# ##################################################################
# sanitize voice spec
# ensure every referenced voice id is real; fall back gracefully
def _sanitize(spec: str, fallback: str) -> str:
    spec = (spec or "").strip()
    if not spec:
        return fallback
    parts = []
    for part in spec.split("+"):
        name, _, w = part.partition("*")
        name = name.strip()
        if name not in ALL_VOICES:
            return fallback
        parts.append(f"{name}*{w.strip()}" if w.strip() else name)
    return "+".join(parts)


# ##################################################################
# deterministic validation + bounded repair
def validate_against_plan(mapping: dict, plan: dict, pool: list[str]) -> tuple[dict, list[str]]:
    """Hard checks against the fixed bank; bounded deterministic repair.

    Checks: every spec resolves to real ids; the bank gender of every used
    voice matches the plan gender; blend components share one gender and one
    accent prefix; cohort members' voices share one accent prefix; no two
    characters share an identical spec+speed when their plan says they must be
    distinct.

    Repair is bounded and deterministic: an offending spec is replaced by the
    first unused (or same-accent-pool) bank voice that satisfies the profile.
    When no candidate exists the mismatch FAILS CLOSED (returned as a
    violation; the caller decides to raise).
    """
    problems: list[str] = []
    repaired: dict[str, dict] = {}
    profiles = plan.get("characters", {})
    used_specs: dict[str, str] = {}
    used_voices: set[str] = set()

    def bank_gender(voice_id: str) -> str:
        return "female" if voice_id[1] == "f" else "male"

    def candidate_for(char_id: str, profile: dict | None, exclude: set[str], prefix: str | None = None) -> str | None:
        wanted = profile.get("gender") if profile else None
        if prefix is None:
            prefix = accent_prefix(profile.get("accent", "")) if profile else None
        candidates = list(pool)
        size = (profile or {}).get("vocal_size")
        if size in ("rich", "light"):
            candidates.sort(key=lambda voice: F0_MEDIANS.get(voice, float("inf")), reverse=size == "light")
        for voice in candidates:
            if voice in exclude:
                continue
            if prefix and voice[0] != prefix:
                continue
            if wanted in ("male", "female") and bank_gender(voice) != wanted:
                continue
            return voice
        return None

    for cid, entry in sorted(mapping.items()):
        if str(cid).startswith("_"):
            continue
        spec = entry.get("voice", "")
        speed = entry.get("speed", 1.0)
        profile = profiles.get(cid)
        parts = voice_parts(spec)
        ids = [name for name, _ in parts]
        valid_ids = all(name in ALL_VOICES and name in pool for name in ids) and bool(ids) and all(math.isfinite(weight) and weight > 0 for _, weight in parts)
        violations: list[str] = []

        if valid_ids:
            genders = {bank_gender(v) for v in ids}
            prefixes = {v[0] for v in ids}
            if len(genders) > 1:
                violations.append(f"{cid}: blend {spec} mixes genders {sorted(genders)}")
            if len(prefixes) > 1:
                violations.append(f"{cid}: blend {spec} mixes accents {sorted(prefixes)}")
            if profile:
                wanted = profile.get("gender")
                if wanted in ("male", "female") and genders != {wanted}:
                    violations.append(
                        f"{cid}: plan gender {wanted} but voice {spec} is {sorted(genders)}"
                    )
                prefix = accent_prefix(profile.get("accent", ""))
                if prefix and prefixes != {prefix}:
                    violations.append(
                        f"{cid}: plan accent '{profile['accent']}' needs prefix '{prefix}' but {spec} uses {sorted(prefixes)}"
                    )
                measured = blend_f0(spec)
                peers = [F0_MEDIANS[voice] for voice in pool if voice in F0_MEDIANS and (not prefix or voice[0] == prefix) and (wanted not in ("male", "female") or bank_gender(voice) == wanted)]
                if measured is not None and peers:
                    centre = statistics.median(peers)
                    if profile.get("vocal_size") == "rich" and measured > centre:
                        violations.append(f"{cid}: measured pitch {measured:.1f}Hz is too high for the rich vocal profile")
                    elif profile.get("vocal_size") == "light" and measured < centre:
                        violations.append(f"{cid}: measured pitch {measured:.1f}Hz is too low for the light vocal profile")
        else:
            violations.append(f"{cid}: spec {spec!r} is not a bank voice or blend")

        if violations:
            replacement = candidate_for(cid, profile, used_voices)
            if replacement is None:
                # every pool voice of this profile is taken — fall back to the
                # first gender/accent-matching voice even if shared, so repair
                # never fails while valid voices exist
                replacement = candidate_for(cid, profile, set())
            if replacement is None:
                problems.extend(violations)
                problems.append(f"{cid}: no in-pool repair candidate; failing closed")
                repaired[cid] = dict(entry)
                continue
            problems.append(
                f"{cid}: repaired {'; '.join(violations)} → {replacement}"
            )
            spec = replacement
            repaired[cid] = {"voice": spec, "speed": speed}
        else:
            repaired[cid] = {"voice": spec, "speed": speed}

        used_specs[cid] = f"{spec}@{speed}"
        used_voices.update(name for name, _ in voice_parts(spec))
        spec_key = used_specs[cid]
        # distinctness: identical spec+speed for distinct characters is a repair case
        for other, other_key in list(used_specs.items()):
            if other != cid and other_key == spec_key:
                replacement = candidate_for(cid, profile, used_voices)
                if replacement is not None and replacement != spec:
                    problems.append(
                        f"{cid}: duplicate voice spec {spec_key} (shared with {other}) → {replacement}"
                    )
                    spec = replacement
                    repaired[cid] = {"voice": spec, "speed": speed}
                    used_specs[cid] = f"{spec}@{speed}"
                    used_voices.update(name for name, _ in voice_parts(spec))
                else:
                    problems.append(f"{cid}: duplicate voice for {other} cannot be separated in this pool; failing closed")

    # cohort accent coherence over final specs: a repaired mapping already
    # satisfies per-character accent checks, so the cohort check only fires on
    # a mapping whose specs still disagree after repair (unrepaired members).
    from collections import Counter

    for cohort in plan.get("cohorts", []):
        exceptions = set(cohort.get("accent_exceptions", []))
        members = [m for m in cohort["members"] if m in repaired and m not in exceptions]
        prefixes = set()
        for member in members:
            for name, _ in voice_parts(repaired[member]["voice"]):
                prefixes.add(name[0])
        if len(prefixes) > 1:
            # bounded deterministic cohort repair: move every member off the
            # minority prefixes onto the majority prefix before failing closed
            counts = Counter(repaired[m]["voice"][0] for m in members)
            majority = counts.most_common(1)[0][0]
            for member in members:
                spec = repaired[member]["voice"]
                if spec[0] != majority:
                    replacement = candidate_for(
                        member,
                        profiles.get(member),
                        used_voices | {spec},
                        prefix=accent_prefix(cohort["accent"]),
                    )
                    if replacement is None:
                        problems.append(
                            f"cohort {cohort['id']} spans accent prefixes {sorted(prefixes)} "
                            f"(accent '{cohort['accent']}') and {member} has no in-pool repair; failing closed"
                        )
                        continue
                    problems.append(
                        f"cohort {cohort['id']}: {member} {spec} → {replacement} for shared accent"
                    )
                    repaired[member]["voice"] = replacement
                    used_specs[member] = f"{replacement}@{repaired[member].get('speed', 1.0)}"
                    used_voices.update(name for name, _ in voice_parts(replacement))
    return repaired, problems


# ##################################################################
# measured pitch helper
def blend_f0(spec: str) -> float | None:
    """Weighted mean f0 of a blend from the measured table, when known."""
    parts = voice_parts(spec)
    known = [(F0_MEDIANS[name], w) for name, w in parts if name in F0_MEDIANS]
    if not known or len(known) != len(parts):
        return None
    total_w = sum(w for _, w in known)
    return sum(f0 * w for f0, w in known) / total_w if total_w else None


# ##################################################################
# map characters to voices
# produce kokoro_voices.json from voices.json using one LLM call
def map_characters_to_voices(output_dir: Path, voices: list[str] | None = None, cache_root: Path | str | None = None) -> Path:
    voices_path = output_dir / "voices.json"
    plan_path = output_dir / "cast_plan.json"
    out_path = output_dir / "kokoro_voices.json"
    from src.cast_plan import SIDECAR_KOKORO, artifact_matches_plan

    plan = None
    pool = voices or ENGLISH_VOICES
    if plan_path.is_file():
        from src.voice_assets import require_current_descriptions

        plan = require_current_descriptions(output_dir)
    elif (output_dir / SIDECAR_KOKORO).is_file():
        raise ValueError(f"casting plan missing for existing Kokoro mapping in {output_dir}")
    # an existing mapping is authoritative ONLY while its fingerprint sidecar
    # matches the current plan; plan drift forces regeneration (no silent
    # stale-map reuse)
    if out_path.exists():
        if plan is not None and artifact_matches_plan(output_dir, SIDECAR_KOKORO, plan):
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if all(voice_parts(entry["voice"]) and all(name in pool for name, _ in voice_parts(entry["voice"])) for entry in cached.values()):
                return out_path
        if plan is None:
            return out_path  # legacy projects without plans keep their explicit mapping
        if plan is not None:
            print("  cast plan changed — regenerating kokoro mapping (was stale)")
    if not voices_path.exists():
        raise ValueError("voices.json not found — run voices_desc step first")
    characters = json.loads(voices_path.read_text(encoding="utf-8"))

    # Unsupported accents fail closed with an actionable recommendation: the
    # English kokoro bank has no honestly French-accented (etc.) English voice —
    # ff_siwis is a French-language pack, not French-accented English — so we
    # refuse to silently Americanize those characters.
    if plan:
        unsupported = [
            (cid, entry["accent"])
            for cid, entry in plan.get("characters", {}).items()
            if accent_prefix(entry.get("accent", "")) not in {voice[0] for voice in pool}
            and entry.get("accent") not in ("", "unknown", None)
        ]
        if unsupported:
            names = ", ".join(f"{cid} ({accent})" for cid, accent in unsupported)
            raise ValueError(
                "kokoro cannot honestly voice non-English accents with its English bank — "
                f"{names}. Use Breeze or Qwen (local/config.toml [tts].engine), which design "
                "voices from descriptions, instead of silently Americanizing these characters."
            )

    raw = ask_sync(_build_prompt(characters, pool, plan), temperature=0.4, max_tokens=4096, root=cache_root)
    try:
        mapping = _parse_mapping(raw)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("Kokoro returned a malformed cast mapping") from error
    if not isinstance(mapping, dict) or set(mapping) != set(characters):
        raise ValueError("Kokoro mapping identities differ from the supplied cast; refusing default voices")

    result: dict[str, dict] = {}
    for cid, info in characters.items():
        if str(cid).startswith("_"):
            continue
        entry = mapping.get(cid, {}) if isinstance(mapping, dict) else {}
        fallback = DEFAULT_NARRATOR if cid == "narrator" else DEFAULT_VOICE
        voice = _sanitize(entry.get("voice", ""), fallback)
        try:
            speed = float(entry.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.7, min(1.3, speed))
        result[cid] = {
            "voice": voice,
            "speed": round(speed, 3),
            "description": info.get("description") or info.get("bio") or "",
        }

    if plan:
        repaired, problems = validate_against_plan(result, plan, pool)
        if problems:
            print("  kokoro cast-plan validation:")
            for problem in problems:
                print(f"    - {problem}")
            fatal = [p for p in problems if "failing closed" in p]
            if fatal:
                raise ValueError(
                    "kokoro mapping incompatible with cast plan (no repair available): "
                    + "; ".join(fatal)
                )
            result = repaired
        # record measured pitch where known for downstream distinctness checks
        for cid, entry in result.items():
            f0 = blend_f0(entry["voice"])
            if f0 is not None:
                entry["f0_median_hz"] = round(f0, 1)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if plan:
        from src.cast_plan import write_plan_sidecar

        write_plan_sidecar(output_dir, plan["fingerprint"], SIDECAR_KOKORO)
    return out_path


# ##################################################################
# load voice map
# read kokoro_voices.json → {char_id: (voice_spec, speed)}
def load_voice_map(output_dir: Path) -> dict[str, tuple[str, float]]:
    p = output_dir / "kokoro_voices.json"
    if not p.exists():
        raise ValueError("kokoro_voices.json not found — run voice mapping first")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {cid: (v["voice"], float(v.get("speed", 1.0))) for cid, v in data.items()}
