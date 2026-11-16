"""Whole-cast voice plan: profile consistency and accent coherence.

One LLM call over the full cast replaces per-character parallel guessing for
the *profile* layer: gender, age band, accent, vocal size, distinctness. The
plan is persisted to ``cast_plan.json`` with an inputs fingerprint
(characters.json digest + per-character source-context digests) so every
consumer — voice descriptions, kokoro mapping, acoustic tightening — works
from the same evidence, and a changed cast plan invalidates derived caches
instead of silently reusing them.

Evidence model
--------------
Every profile fact carries ``{"fact": ..., "source": ...}`` with source ranked:
``metadata`` (importer-supplied verbatim context) > ``bio`` (characters.json
voice notes) > ``setting`` (whole-book context) > ``inferred`` (model
reasoning from documented cues). Facts the evidence does not support stay
``unknown`` — demographics are never guessed from names, and ethnicity/race is
recorded as evidence only: it never determines pitch, accent, timbre, or any
other acoustic property.

Cohorts
-------
People who share an upbringing, place, or household share the same accent
unless their own evidence shows migration (an explicit individual accent beats
the cohort default). Independent voices inside a shared accent are preserved
by the ``distinct_from`` constraint (pitch/timbre separation), never by
changing the accent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from src.casting_llm import ask_cached_async as ask

PLAN_VERSION = 3
PLAN_NAME = "cast_plan.json"
# Sidecar that binds a derived artifact (voices.json, kokoro_voices.json) to
# the cast-plan fingerprint it was produced against, WITHOUT adding keys to
# the artifact itself — consumers iterate those files and must never see a
# metadata key where a character belongs.
SIDECAR_VOICES = "voices_plan.json"
SIDECAR_KOKORO = "kokoro_voices_plan.json"
GENDERS = ("male", "female", "nonbinary", "unknown")
AGE_BANDS = ("child", "teen", "young_adult", "adult", "middle_aged", "older", "unknown")
SOURCES = ("metadata", "bio", "setting", "inferred")
COHORT_KINDS = ("upbringing", "place", "household")
# Per-character source-context budget: the whole-cast call must fit the
# primary 32768-token ctx with room for the output, without ever dropping
# explicit importer evidence.
MAX_CONTEXT_CHARS = 2400
# Whole-prompt bound: bios plus contexts plus prompt scaffolding, sized for a
# 10-speaker novel whose combined importer bios run ~41k chars.
MAX_PROMPT_CHARS = 60_000
# Per-character bio bound inside the joint call (importer bios repeat premise;
# explicit demographic/upbringing lines are never dropped).
MAX_BIO_CHARS = 6000

VOCAL_SIZES = ("light", "medium", "rich", "unknown")


# ##################################################################
# digests
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def characters_digest(characters: dict) -> str:
    return _sha(json.dumps(characters, sort_keys=True, ensure_ascii=False))


def _context_digest(project: Path, char_id: str) -> str:
    """Digest of the importer-supplied verbatim context for one character.

    Two supported shapes: the direct-import ``source_context.json`` map
    (char_id → excerpt string or {"text"|"excerpt"|"context": ...}) and the
    per-file ``source_context/<char_id>.txt`` directory. Both are optional:
    pure-book projects have neither.
    """
    path = project / "source_context" / f"{char_id}.txt"
    if path.is_file():
        return _sha(path.read_text(encoding="utf-8"))
    return ""


def _load_source_context_map(project: Path) -> dict[str, str]:
    source = project / "source_context.json"
    if not source.is_file():
        return {}
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for char_id, value in data.items():
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            text = str(value.get("text") or value.get("excerpt") or value.get("context") or "")
        else:
            text = ""
        if text.strip():
            out[str(char_id)] = text
    return out


# ##################################################################
# cast input
# bounded per-character cast input for the joint call; importer context is
# trimmed to budget without dropping its explicit demographic/upbringing lines.
def _prioritized_context(raw: str) -> str:
    if len(raw) <= MAX_CONTEXT_CHARS:
        return raw
    keep = [
        line
        for line in raw.splitlines()
        if re.search(
            r"(?i)\b(age|aged|years? old|gender|man|woman|male|female|boy|girl|"
            r"grew up|raised|upbringing|accent|speaks?|nationality|from|lives? in|"
            r"household|family|brother|sister|son|daughter|mother|father)\b",
            line,
        )
    ]
    if keep and sum(len(line) for line in keep) <= MAX_CONTEXT_CHARS:
        return "\n".join(keep)
    return raw[:MAX_CONTEXT_CHARS]


def _prioritized_bio(raw: str) -> str:
    """Bounded bio: kept whole under budget; over budget, explicit
    demographic/upbringing lines are kept and only narrative filler is cut —
    never the reverse."""
    if len(raw) <= MAX_BIO_CHARS:
        return raw
    keep = [
        line
        for line in raw.splitlines()
        if re.search(
            r"(?i)\b(age|aged|years? old|gender|man|woman|male|female|boy|girl|"
            r"grew up|raised|upbringing|accent|speaks?|nationality|from|lives? in|"
            r"household|family|brother|sister|son|daughter|mother|father)\b",
            line,
        )
    ]
    if keep and sum(len(line) for line in keep) <= MAX_BIO_CHARS:
        return "\n".join(keep)
    return raw[:MAX_BIO_CHARS]


def cast_input(characters: dict, contexts: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for char_id, info in characters.items():
        context = _prioritized_context((contexts.get(char_id) or "").strip())
        profile = info.get("profile") if isinstance(info.get("profile"), dict) else {}
        evidence = info.get("evidence") if isinstance(info.get("evidence"), list) else []
        rows.append(
            {
                "id": char_id,
                "name": str(info.get("name", char_id)),
                "bio": _prioritized_bio(str(info.get("bio") or info.get("description") or "")),
                "context": context,
                "profile": profile,
                "evidence": evidence,
            }
        )
    return rows


def _load_contexts(project: Path, characters: dict) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for char_id, text in _load_source_context_map(project).items():
        if char_id in characters:
            contexts[char_id] = text
    ctx_dir = project / "source_context"
    if ctx_dir.is_dir():
        for char_id in characters:
            path = ctx_dir / f"{char_id}.txt"
            if path.is_file() and char_id not in contexts:
                contexts[char_id] = path.read_text(encoding="utf-8")
    return contexts


# ##################################################################
# fingerprint
def plan_fingerprint(characters: dict, contexts: dict[str, str], story_context: str = "") -> dict:
    """Full inputs fingerprint persisted with the plan (stale-cache key)."""
    return {
        "plan_version": PLAN_VERSION,
        "characters_sha256": characters_digest(characters),
        "story_sha256": _sha(story_context),
        "context_sha256": {
            char_id: _sha(text) for char_id, text in sorted(contexts.items()) if text
        },
    }


# ##################################################################
# prompt
def _cast_block(rows: list[dict]) -> str:
    parts: list[str] = []
    for row in rows:
        segment = [f'### {row["id"]} (name: {row["name"]})']
        if row["context"]:
            segment.append(f"IMPORTER CONTEXT (verbatim, highest authority):\n{row['context']}")
        segment.append(f"BIO:\n{row['bio'] or '(none)'}")
        parts.append("\n".join(segment))
    return "\n\n".join(parts)


def build_prompt(rows: list[dict], story_context: str) -> str:
    story = (story_context or "").strip()
    if len(story) > 4000:
        story = story[:4000]
    cast_text = _cast_block(rows)
    return f"""You are the casting director for an audiobook. Design ONE coherent voice plan for the WHOLE cast below in a single pass.

STORY CONTEXT (whole-book premise, shared by everyone):
{story or "(none supplied)"}

CAST (each entry: importer context where supplied, then character bio):
{cast_text}

For EVERY character decide:
- "gender": male | female | nonbinary | unknown — only from pronouns/descriptions/importer context. NEVER from a name.
- "age_band": child | teen | young_adult | adult | middle_aged | older | unknown — from stated age or documented cues (era, occupation, life stage). War/young-sounding settings alone do NOT imply young. If nothing supports an age, use unknown.
- "approx_age": short string ("mid 30s") or null.
- "accent": the accent they speak with NOW, e.g. "American English", "British English (South London)", "Italian-accented English". Use importer context first, then bio, then setting cues. "unknown" only if no evidence exists.
- "vocal_size": light | medium | rich | unknown — perceived vocal size/depth (a large heavyset man is usually rich; a small frail elder light).
- "distinct_from": list of other character ids whose voices must sound clearly different from this one (same gender/accent peers first).
- "evidence": list {{"fact": "...", "source": "metadata|bio|setting|inferred"}} — every profile fact above must appear here with the best source you have. Race/ethnicity may be recorded as evidence ONLY; it must never determine gender, pitch, accent, or vocal_size.

COHORTS — accent coherence (formative speech communities, not current residence):
- Group characters who canonically share a FORMATIVE speech community into cohorts — same upbringing, same childhood place, same origin culture: {{"id": "...", "kind": "upbringing|place|household", "members": [ids], "accent": "...", "basis": "why"}}.
- "place" means a shared formative place (where accents are acquired), not merely living in the same city now. Adults who moved somewhere together RETAIN the accent of where they grew up — cohort them by upbringing/origin, not current address. "household" is for people raised in the same household.
- Cohort members share the cohort accent UNLESS their own explicit evidence shows a different personal accent (e.g. grew up elsewhere, individual migration) — an individual accent beats the cohort default.
- People with DIFFERENT backgrounds legitimately have different accents: never equalize accents across separate cohorts or for characters whose evidence differs.
- Explicit individual migration/accent evidence always wins over cohort default.

Rules:
- Use ONLY the evidence given. Missing evidence = "unknown", never a guess. Do NOT infer any demographic from a character's NAME or from race/ethnicity.
- Keep narrator gender/age/accent only if evidence exists; otherwise unknown (no default narrator voice).

Output ONLY valid JSON, no markdown:
{{
  "characters": {{
    "<char_id>": {{
      "gender": "...", "age_band": "...", "approx_age": null, "accent": "...",
      "vocal_size": "...", "distinct_from": [], "accent_source": "individual|cohort",
      "evidence": [{{"fact": "...", "source": "metadata"}}]
    }}
  }},
  "cohorts": [
    {{"id": "...", "kind": "upbringing", "members": ["a", "b"], "accent": "...", "basis": "..."}}
  ]
}}"""


# ##################################################################
# parsing / normalization
def _norm_choice(value, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _as_id_list(value, known: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: list[str] = []
    for item in value:
        text = str(item).strip()
        if text in known and text not in seen:
            seen.append(text)
    return seen


def _norm_evidence(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        source = _norm_choice(item.get("source"), SOURCES, "inferred")
        if fact:
            out.append({"fact": fact[:300], "source": source})
    return out


def normalize_plan(raw: dict, known: set[str], rows: list[dict]) -> dict:
    """Typed, bounded plan from the model response; identity set must match.

    A response that omits requested identities or invents unknown ones is a
    malformed answer, not an all-unknown plan: nothing is silently degraded.
    """
    characters_in = raw.get("characters") if isinstance(raw.get("characters"), dict) else {}
    missing = sorted(known - set(characters_in))
    extra = sorted(set(characters_in) - known)
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing identities: {missing}")
        if extra:
            detail.append(f"unknown identities: {extra}")
        raise ValueError("cast plan identity mismatch — " + "; ".join(detail))
    characters: dict[str, dict] = {}
    for row in rows:
        char_id = row["id"]
        entry = characters_in.get(char_id, {}) if isinstance(characters_in.get(char_id), dict) else {}
        characters[char_id] = {
            "name": row["name"],
            "gender": _norm_choice(entry.get("gender"), GENDERS, "unknown"),
            "age_band": _norm_choice(entry.get("age_band"), AGE_BANDS, "unknown"),
            "approx_age": (str(entry["approx_age"]).strip()[:40] or None)
            if entry.get("approx_age")
            else None,
            "accent": str(entry.get("accent") or "").strip()[:120] or "unknown",
            "vocal_size": _norm_choice(entry.get("vocal_size"), VOCAL_SIZES, "unknown"),
            "distinct_from": _as_id_list(entry.get("distinct_from"), known),
            "accent_source": "cohort" if str(entry.get("accent_source")) == "cohort" else "individual",
            "evidence": _norm_evidence(entry.get("evidence"))[:12],
        }
    cohorts: list[dict] = []
    for group in raw.get("cohorts") if isinstance(raw.get("cohorts"), list) else []:
        if not isinstance(group, dict):
            continue
        members = _as_id_list(group.get("members"), known)
        if len(members) < 2:
            continue
        cohorts.append(
            {
                "id": str(group.get("id") or "_".join(members))[:80],
                "kind": _norm_choice(group.get("kind"), COHORT_KINDS, "upbringing"),
                "members": members,
                "accent": str(group.get("accent") or "").strip()[:120] or "unknown",
                "basis": str(group.get("basis") or "").strip()[:300],
            }
        )
    return {"characters": characters, "cohorts": cohorts}


def parse_plan(text: str) -> dict:
    """Tolerant JSON extraction (mirrors repo convention)."""
    text = text.strip()
    if "```" in text:
        start, end = text.find("```"), text.rfind("```")
        if start != end:
            lines = text[start : end + 3].split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1:
            text = text[i : j + 1]
    return json.loads(text)


def _explicit_accent_entry(entry: dict) -> bool:
    """The model explicitly marked an individual accent distinct from its cohort,
    backed by accent/migration evidence."""
    return entry.get("accent_source") == "individual" and bool(
        re.search(r"(?i)accent|grew up|raised|migrat|moved|from", " ".join(e["fact"] for e in entry.get("evidence", [])))
    )


# ##################################################################
# cohort resolution
def resolve_cohort_accents(plan: dict) -> dict:
    """Apply cohort accents to members lacking an explicit individual accent.

    A member keeps its own accent only when the model marked it
    ``accent_source: individual`` AND it differs from the cohort accent AND the
    member's evidence mentions an accent/migration fact; otherwise the cohort
    accent wins (coherence without destroying explicit exceptions). The
    preserved-exception set is recorded on the cohort so validation can exempt
    exactly those members.
    """
    characters = plan["characters"]
    for cohort in plan["cohorts"]:
        exceptions: list[str] = []
        for member in cohort["members"]:
            entry = characters.get(member)
            if entry is None:
                continue
            own = entry["accent"]
            if (
                own != "unknown"
                and own != cohort["accent"]
                and _explicit_accent_entry(entry)
            ):
                exceptions.append(member)
                continue  # explicit individual accent preserved
            entry["accent"] = cohort["accent"]
        cohort["accent_exceptions"] = exceptions
    return plan


# ##################################################################
# validation
def validate_plan(plan: dict) -> list[str]:
    """Deterministic consistency checks; returns human-readable violations."""
    problems: list[str] = []
    characters = plan.get("characters", {})
    for cohort in plan.get("cohorts", []):
        exceptions = set(cohort.get("accent_exceptions", []))
        accents = {
            characters[m]["accent"]
            for m in cohort["members"]
            if m in characters and m not in exceptions
        }
        if len(accents) > 1:
            problems.append(
                f"cohort {cohort['id']} members have inconsistent accents: {sorted(accents)}"
            )
    for char_id, entry in characters.items():
        if entry["gender"] not in GENDERS or entry["age_band"] not in AGE_BANDS:
            problems.append(f"{char_id} has invalid gender/age_band")
        if entry["vocal_size"] not in VOCAL_SIZES:
            problems.append(f"{char_id} has invalid vocal_size")
    return problems


# ##################################################################
# staleness
def plan_is_stale(project: Path, characters: dict, contexts: dict[str, str] | None = None) -> bool:
    """True when no plan exists or its fingerprint no longer matches inputs."""
    plan_path = project / PLAN_NAME
    if not plan_path.is_file():
        return True
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    if plan.get("fingerprint", {}).get("plan_version") != PLAN_VERSION:
        return True
    if contexts is None:
        contexts = _load_contexts(project, characters)
    story = source_story_context(project) or str(plan.get("story_context") or "")
    return plan.get("fingerprint") != plan_fingerprint(characters, contexts, story)


def plan_missing_characters(project: Path, characters: dict) -> list[str]:
    """Characters present in characters.json but absent from the persisted plan."""
    plan_path = project / PLAN_NAME
    if not plan_path.is_file():
        return sorted(characters)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return sorted(characters)
    have = plan.get("characters", {})
    return sorted(cid for cid in characters if cid not in have)


# ##################################################################
# plan or refresh
async def build_cast_plan(
    project: Path,
    characters: dict | None = None,
    story_context: str = "",
    cache_root: Path | str | None = None,
) -> dict:
    """Load a fresh plan, or regenerate when the inputs fingerprint changed.

    Fail-closed: a plan whose regenerated form still fails deterministic
    validation raises rather than persisting an inconsistent cast.
    """
    characters_path = project / "characters.json"
    if characters is None:
        characters = json.loads(characters_path.read_text(encoding="utf-8"))
    contexts = _load_contexts(project, characters)
    if not story_context:
        story_context = source_story_context(project)
        previous = project / PLAN_NAME
        if not story_context and previous.is_file():
            story_context = str(load_plan(project).get("story_context") or "")
    prompt = build_prompt(cast_input(characters, contexts), story_context)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"cast plan prompt {len(prompt)} chars exceeds bound {MAX_PROMPT_CHARS}; "
            "reduce source context"
        )
    raw_text = (await ask(prompt, temperature=0.2, max_tokens=8192, root=cache_root)).strip()
    raw = parse_plan(raw_text)
    known = set(characters)
    plan = normalize_plan(raw, known, cast_input(characters, contexts))
    resolve_cohort_accents(plan)
    problems = validate_plan(plan)
    if problems:
        raise ValueError("cast plan validation failed: " + "; ".join(problems))
    document = {
        "fingerprint": plan_fingerprint(characters, contexts, story_context),
        "story_context": story_context,
        "prompt_chars": len(prompt),
        "characters": plan["characters"],
        "cohorts": plan["cohorts"],
    }
    (project / PLAN_NAME).write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return document


def build_cast_plan_sync(project: Path, characters: dict | None = None, story_context: str = "", cache_root: Path | str | None = None) -> dict:
    return asyncio.run(build_cast_plan(project, characters, story_context, cache_root))


def load_plan(project: Path) -> dict:
    plan_path = project / PLAN_NAME
    if not plan_path.is_file():
        raise ValueError(f"{PLAN_NAME} not found — build the cast plan first")
    return json.loads(plan_path.read_text(encoding="utf-8"))


# ##################################################################
# fingerprint sidecars
def read_plan_sidecar(project: Path, name: str) -> dict | None:
    path = project / name
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def source_story_context(project: Path) -> str:
    path = project / "source_context.json"
    if not path.is_file():
        return ""
    value = json.loads(path.read_text(encoding="utf-8"))
    return str(value.get("description") or "") if isinstance(value, dict) else ""


def plan_content_digest(plan: dict) -> str:
    return _sha(json.dumps({"characters": plan["characters"], "cohorts": plan.get("cohorts", [])}, sort_keys=True))


def artifact_matches_plan(project: Path, name: str, plan: dict) -> bool:
    sidecar = read_plan_sidecar(project, name)
    artifact = project / ("voices.json" if name == SIDECAR_VOICES else "kokoro_voices.json")
    return bool(
        sidecar and artifact.is_file()
        and sidecar.get("fingerprint") == plan.get("fingerprint")
        and sidecar.get("plan_sha256") == plan_content_digest(plan)
        and sidecar.get("artifact_sha256") == hashlib.sha256(artifact.read_bytes()).hexdigest()
        and (name != SIDECAR_KOKORO or sidecar.get("voices_sha256") == hashlib.sha256((project / "voices.json").read_bytes()).hexdigest())
    )


def write_plan_sidecar(project: Path, fingerprint: dict, name: str) -> None:
    plan = load_plan(project)
    artifact = project / ("voices.json" if name == SIDECAR_VOICES else "kokoro_voices.json")
    document = {
        "fingerprint": fingerprint,
        "plan_sha256": plan_content_digest(plan),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    if name == SIDECAR_KOKORO:
        document["voices_sha256"] = hashlib.sha256((project / "voices.json").read_bytes()).hexdigest()
    (project / name).write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


# ##################################################################
# profile block for consumers
def profile_block(plan: dict, char_id: str) -> str:
    """Compact constraint text for one character, embedded in consumer prompts."""
    entry = plan["characters"].get(char_id)
    if entry is None:
        return ""
    lines = [
        f"Gender: {entry['gender']}",
        f"Age: {entry['age_band']}" + (f" (about {entry['approx_age']})" if entry["approx_age"] else ""),
        f"Accent: {entry['accent']}",
        f"Vocal size: {entry['vocal_size']}",
    ]
    if entry["distinct_from"]:
        lines.append(f"Must sound clearly distinct from: {', '.join(entry['distinct_from'])}")
    facts = [e["fact"] for e in entry["evidence"]][:6]
    if facts:
        lines.append("Established facts: " + "; ".join(facts))
    return "\n".join(lines)
