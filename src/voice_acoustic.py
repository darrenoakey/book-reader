import json
import re
from pathlib import Path

from src.casting_llm import ask_cached_async

SYSTEM_PROMPT = """You convert character biographies into terse acoustic voice descriptions for the qwen3-tts voice-design model.

Rules from the qwen3-tts documentation:
- 1 to 3 sentences, ideally 15-40 words total.
- Describe ACOUSTIC qualities only: gender, age, pitch, timbre, accent, pace, emotion, distinctive vocal traits.
- DO NOT include personality, biography, plot role, relationships, philosophy, what they say, or their background story.
- Be specific: "deep", "crisp", "fast-paced", "soft and breathy" — NOT "nice", "sophisticated".
- Combine multiple dimensions: gender + age + accent + pitch + timbre + pace.
- AVOID conflicting attributes (e.g. "high-pitched deep bass" — pick one).
- No celebrity names.
- English text only. The output speech may be in any language but the instruct text must be English.
- Never derive pitch, accent or timbre from race or ethnicity.

When a CASTING PLAN PROFILE is supplied it is authoritative: the description
MUST state the profile's gender, age band and accent (a profile "unknown"
field is freedom — omit it, never invent a value).

You will be given one character's bio at a time. Reply with ONLY the acoustic description — no preamble, no quotes, no explanation. Just the description text."""

# Deterministic profile→acoustic-keyword expectations: a description that
# contradicts the persisted plan (wrong gender words, wrong accent words,
# conflicting age cues) is repaired by re-asking once with the contradiction
# named; a still-contradicting description fails closed.
_GENDER_WORDS = {
    "male": {"male", "masculine", "gentleman"},
    "female": {"female", "feminine", "lady"},
}
# Stated positively (plan gender MUST appear) — substring traps like
# "woman"/"man" inside other words are checked word-boundary-wise below.
_GENDER_POSITIVE = {
    "male": ("male", "man", "masculine", "gentleman", "fellow", "he"),
    "female": ("female", "woman", "feminine", "lady", "she"),
}
_AGE_CUES = {
    "child": ("child", "young child", "kid", "girl's", "boy's"),
    "teen": ("teenage", "teen", "adolescent", "youthful"),
    "young_adult": ("young adult", "twenties", "early thirties", "youthful", "young"),
    "adult": ("adult", "thirties", "forties"),
    "middle_aged": ("middle-aged", "middle aged", "fifties", "forties"),
    "older": ("older", "elderly", "sixties", "seventies", "old man", "old woman"),
}
_ACCENT_HINTS = {
    "british": ("british", "rp", "london", "scottish", "welsh", "irish", "uk"),
    "american": ("american", "midwest", "californian", "new york"),
    "spanish": ("spanish", "castilian"),
    "french": ("french", "parisian"),
    "italian": ("italian",),
    "australian": ("australian", "aussie"),
    "indian": ("indian", "hindi"),
}


# ##################################################################
# deterministic check
# returns the list of contradictions between a description and a plan profile
def _mentions(text_lower: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text_lower) is not None


# Opposite-gender word forms that are unambiguous on their own (word-boundary
# matched): only these trigger a contradiction, avoiding the woman/man
# substring overlap ("woman" contains "man").
_OPPOSITE_FORMS = {
    "male": ("woman", "female", "feminine", "lady", "she"),
    "female": ("man", "male", "masculine", "gentleman", "he"),
}


# ##################################################################
# self-only voice direction
# A comparison with another actor is not a trait of this actor. Remove those
# clauses from the actual TTS instruction as well as validation, so a line
# such as 'distinct from French-accented characters' cannot recolor an American.
def self_description(description: str) -> str:
    parts = []
    marker = re.compile(r"\b(?:that\s+distinguish\w*|distinguish\w*|differentiat\w*|distinct(?:ly)?\s+from|different\s+from|contrasts?\s+with|in contrast|unlike|rather than|compared (?:with|to)|without|avoid(?:ing)?|not)\b", re.IGNORECASE)
    for sentence in re.split(r"(?<=[.!?])\s+", description.strip()):
        match = marker.search(sentence)
        if match:
            sentence = sentence[:match.start()].rstrip(" ,;:-")
        if sentence:
            parts.append(sentence if sentence[-1] in ".!?" else sentence + ".")
    return " ".join(parts)


_DECADES = {"twenties": 20, "thirties": 30, "forties": 40, "fifties": 50, "sixties": 60, "seventies": 70, "eighties": 80, "nineties": 90}


def description_violations(description: str, profile: dict) -> list[str]:
    problems: list[str] = []
    text = self_description(description).lower()
    gender = profile.get("gender")
    if gender in ("male", "female"):
        stated = any(_mentions(text, w) for w in _GENDER_POSITIVE[gender])
        if not stated:
            problems.append(f"description does not state gender '{gender}'")
        if any(_mentions(text, w) for w in _OPPOSITE_FORMS[gender]):
            problems.append(f"description implies the opposite gender; plan says '{gender}'")
    age_band = profile.get("age_band")
    if age_band and age_band != "unknown":
        cues = _AGE_CUES.get(age_band, ())
        numeric_ages = [int(value) for value in re.findall(r"\b(\d{1,3})[- ]years?[- ]old\b", text)]
        # decade words are ages too: "in her fifties or sixties" is checked against the band
        # (a middle-aged 50s-60s woman is not "elderly"; a middle-aged "seventies" still fails)
        numeric_ages += [_DECADES[word] for word in re.findall(r"\b(" + "|".join(_DECADES) + r")\b", text)]
        # "in her early 20s" is an age too, not just "22 years old" or "twenties"
        numeric_ages += [int(value) for value in re.findall(r"\b(?:age[d]?|about|around|early|mid|late)\s+(\d{1,3})s?\b", text)]
        ranges = {"child": (0, 12), "teen": (13, 19), "young_adult": (20, 34), "adult": (25, 49), "middle_aged": (40, 64), "older": (60, 120)}
        numeric_match = any(ranges[age_band][0] <= age <= ranges[age_band][1] for age in numeric_ages) if age_band in ranges else False
        if cues and not any(_mentions(text, cue) for cue in cues) and not numeric_match:
            problems.append(f"description has no '{age_band}' age cue")
        if numeric_ages and not numeric_match:
            problems.append(f"description states an incompatible numeric age for '{age_band}'")
        hard_age_cues = {
            "child": ("young child", "child voice", "child's voice", "little boy", "little girl"),
            "older": ("elderly", "old man", "old woman", "older adult", "older male", "older female", "older man", "older woman", "older voice"),
        }
        for other, cues_other in hard_age_cues.items():
            # Job seniority and relative age ('older sister') do not mean elderly.
            if other != age_band and any(_mentions(text, cue) for cue in cues_other):
                problems.append(f"description implies age '{other}', plan says '{age_band}'")
    accent = (profile.get("accent") or "").lower()
    if accent and accent != "unknown":
        for needle, hints in _ACCENT_HINTS.items():
            if needle in accent:
                if not any(_mentions(text, h) for h in hints):
                    problems.append(f"description does not carry the '{accent}' accent")
                for other, other_hints in _ACCENT_HINTS.items():
                    if other != needle and any(re.search(rf"\b{re.escape(h)}(?:[- ]accented|(?: English)? accent)\b", text, re.IGNORECASE) for h in other_hints):
                        problems.append(f"description specifies a conflicting {other} accent")
                break
    if profile.get("vocal_size") == "rich":
        if not any(_mentions(text, word) for word in ("deep", "low", "baritone", "bass", "rich", "resonant", "full-bodied")):
            problems.append("description does not carry the rich/resonant vocal size")
        if re.search(r"(?<!not )(?<!no )(?<!avoiding )\b(?:squeaky|high-pitched)\b", text):
            problems.append("description specifies a thin/high voice contrary to rich vocal size")
    return problems


def _repair_prompt(char_id: str, bio: str, profile: dict, violations: list[str]) -> str:
    lines = [f"Character id: {char_id}", f"Biography:\n{bio}", "", "Your previous reply violated the casting plan:"]
    lines += [f"- {v}" for v in violations]
    lines.append(
        "\nRewrite the acoustic description so it satisfies EVERY plan constraint exactly "
        "(gender, age band, accent). Reply with the description text only."
    )
    return "\n".join(lines)


# ##################################################################
# generate acoustic for one
# one conversation turn with deterministic plan validation and a single
# bounded repair re-ask; still-contradicting output fails closed
async def _acoustic_for_one(char_id: str, bio: str, profile: dict | None) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nCharacter id: {char_id}\nBiography:\n{bio}"
    if profile:
        from src.cast_plan import profile_block

        prompt += f"\n\nCASTING PLAN PROFILE:\n{profile_block({'characters': {char_id: profile}}, char_id)}"
    acoustic = self_description((await ask_cached_async(prompt)).strip())
    violations = description_violations(acoustic, profile) if profile else []
    if not violations:
        return acoustic
    repair = prompt + "\n\nPrevious reply:\n" + acoustic + "\n\n" + _repair_prompt(char_id, bio, profile, violations)
    repaired = self_description((await ask_cached_async(repair)).strip())
    remaining = description_violations(repaired, profile)
    if remaining:
        raise ValueError(f"acoustic description for {char_id} contradicts the plan: {'; '.join(remaining)}")
    return repaired


# ##################################################################
# generate acoustic descriptions
# Use the same verified plan and native LLM boundary as all other casting.
async def generate_acoustic_descriptions(output_dir: Path) -> dict[str, str]:
    from src.cast_plan import SIDECAR_VOICES, write_plan_sidecar
    from src.voice_assets import require_current_descriptions

    voices_path = output_dir / "voices.json"
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    plan = require_current_descriptions(output_dir)
    profiles = plan["characters"] if plan else {}
    out = {}
    for char_id, info in voices.items():
        acoustic = await _acoustic_for_one(char_id, info.get("description") or info.get("bio") or "", profiles.get(char_id))
        out[char_id] = acoustic
        info["acoustic"] = acoustic
    voices_path.write_text(json.dumps(voices, indent=2, ensure_ascii=False), encoding="utf-8")
    if plan:
        write_plan_sidecar(output_dir, plan["fingerprint"], SIDECAR_VOICES)
    return out


# ##################################################################
# generate one
# Reject stale/missing identity before making any model request.
async def generate_one(output_dir: Path, char_id: str) -> str:
    from src.voice_assets import require_current_descriptions

    voices = json.loads((output_dir / "voices.json").read_text(encoding="utf-8"))
    if char_id not in voices:
        raise ValueError(f"{char_id} not in voices.json")
    plan = require_current_descriptions(output_dir)
    profile = plan["characters"].get(char_id) if plan else None
    return await _acoustic_for_one(char_id, voices[char_id].get("description") or voices[char_id].get("bio") or "", profile)
