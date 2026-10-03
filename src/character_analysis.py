import asyncio
import json
from pathlib import Path

from src.data_recovery import (
    DataIssue,
    OperationalError,
    RecoveryLedger,
    bounded,
    is_data_error,
)
from src.llm import ask

STAGE = "characters"
CHAPTER_WINDOW = 15000
DEFAULT_NARRATOR = {"name": "Narrator", "bio": "A clear, neutral audiobook narrator."}


# ##################################################################
# parse json response
# extract json from claude response handling markdown code blocks and preamble
def parse_json_response(text: str) -> dict:
    parsed = _extract_json(text)
    return {"characters": {}} if parsed is None else parsed


def _extract_json(text: str) -> object | None:
    text = text.strip()
    if not text:
        return None
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
    if not text or not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return None


# ##################################################################
# parse json response strict
# same extraction as parse_json_response but a response that holds no JSON object is a typed DataIssue instead of a silent empty result
def parse_json_response_strict(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise DataIssue("model_response_empty", "model returned an empty response")
    parsed = _extract_json(text)
    if parsed is None:
        raise DataIssue(
            "model_json_unparseable", "model response contains no parseable JSON object", {"response": bounded(text)}
        )
    if not isinstance(parsed, dict):
        raise DataIssue("model_json_not_object", "model JSON is not an object", {"response": bounded(text)})
    return parsed


# ##################################################################
# query haiku
# send a prompt to the LLM (concurrency is bounded inside src.llm.ask)
async def query_haiku(prompt: str) -> str:
    return (await ask(prompt)).strip()


# ##################################################################
# analyze chapter
# extract character information from a single chapter
async def analyze_chapter(chapter_path: Path, chapter_num: int, established_characters: str = "") -> dict:
    text = chapter_path.read_text(encoding="utf-8")
    if not text.strip():
        raise DataIssue("chapter_empty", f"chapter has no text: {chapter_path.name}")
    # No silent truncation: an overlarge chapter is analysed in consecutive windows and merged.
    windows = [text[i : i + CHAPTER_WINDOW] for i in range(0, len(text), CHAPTER_WINDOW)] or [""]
    if len(windows) == 1:
        return await _analyze_text(text, chapter_num, established_characters)
    results = [await _analyze_text(window, chapter_num, established_characters) for window in windows]
    return {"characters": merge_character_info(results)}


async def _analyze_text(text: str, chapter_num: int, established_characters: str) -> dict:
    prompt = f"""Analyze this chapter and identify characters who speak or have internal monologue.

For each speaking character, extract TWO SEPARATE descriptions: how they SOUND (voice) and how they LOOK (look).

VOICE — ALWAYS include nationality/region and a specific age if there are ANY contextual clues (setting, era, vocabulary, place names, period detail) — these are the most important signals for voice. Do not default to "young" or "American" without evidence.
- Gender (from pronouns or descriptions)
- Age — be specific where possible (e.g. "early 30s", "around 50"). Only use "young/middle-aged/elderly" if no clue. War setting alone does NOT mean young.
- Nationality / regional accent (REQUIRED if any contextual evidence exists — WWI British, South African, Egyptian, Australian, etc. Use period and setting cues, not just accent words.)
- Physical build (large, small, thin, heavy, etc.)
- Voice/speech patterns (gruff, soft, educated, crude, accent, lisping, etc.)
- Distinctive physical traits affecting voice (old, frail, booming, wheezing, etc.)

LOOK — harvest EVERY visual detail the text states or directly describes about the character's appearance, exhaustively:
- Hair (color, length, style), eyes (color, shape), face (shape, features, marks, scars, beard)
- Height, build, posture, how they move
- Clothing and its colors, gear, jewelry — exactly as described
- Apparent age as a viewer would see it
- Distinguishing marks a viewer would notice
- Species: state it explicitly (e.g. "ordinary human", "dragon"). NEVER invent animal/dragon/monster features for a character the text describes as a person — humans in the text are plain humans, even in fantasy settings where they bond with dragons.
- Use PLAIN UNIVERSAL visual language. NO in-world proper nouns or setting jargon in the look field — an image generator does not know what "Pernese", "Weyr", or "candidate" means and will guess (badly). Translate into concrete visuals: "boy in a plain white robe", never "Pernese candidate".
- If the text gives NO visual details for a character, say "no visual details given" — do NOT invent an appearance.

Return ONLY valid JSON:
{{
  "characters": {{
    "character_id": {{
      "name": "Display Name",
      "voice": "Physical and voice description only",
      "look": "Every visual detail the text gives, or 'no visual details given'"
    }}
  }}
}}

Rules:
- Include ONLY characters who actually speak (quoted dialogue) or have internal monologue
- Do NOT include characters who are merely mentioned
- Character IDs: lowercase with underscores (e.g., "jean_tannen")
- Established canonical identities (reuse these exact IDs whenever the person is the same): {established_characters or "(none yet)"}
- voice must focus on VOICE generation; look must focus on what a viewer SEES
- EXCLUDE from both: plot roles, story function, relationships to other characters, emotional descriptions
- NO cross-character references (don't mention other characters in the descriptions)

Chapter {chapter_num} text:
{text}"""

    response = await query_haiku(prompt)
    parsed = parse_json_response_strict(response)
    if not isinstance(parsed.get("characters", {}), dict):
        raise DataIssue(
            "model_characters_not_object", "model 'characters' is not an object", {"response": bounded(response)}
        )
    return parsed


# ##################################################################
# merge character info
# combine character info from multiple chapters
def merge_character_info(all_chars: list[dict], ledger: RecoveryLedger | None = None) -> dict:
    merged = {}
    for chapter_index, chapter_chars in enumerate(all_chars):
        entries = chapter_chars.get("characters", {}) if isinstance(chapter_chars, dict) else None
        if not isinstance(entries, dict):
            _warn(
                ledger,
                f"chapter-result-{chapter_index}",
                "chapter_result_malformed",
                "chapter analysis result is not an object with a characters object",
                {"value": bounded(chapter_chars)},
            )
            continue
        for char_id, info in entries.items():
            item = f"chapter-result-{chapter_index}:{bounded(char_id, 80)}"
            if not isinstance(char_id, str) or not char_id.strip() or not isinstance(info, dict):
                _warn(
                    ledger,
                    item,
                    "character_entry_malformed",
                    "character entry has an empty/non-text id or a non-object body",
                    {"id": bounded(char_id), "entry": bounded(info)},
                )
                continue
            # "details" is the legacy single-field shape; "voice"+"look" is the
            # current two-field shape (sound vs appearance kept separate so the
            # movie side never has to mine voice notes for visual facts).
            voice = info.get("voice", info.get("details", info.get("bio", "")))
            look = info.get("look", "")
            name = info.get("name", char_id)
            if not all(isinstance(v, str) for v in (voice, look, name)):
                _warn(
                    ledger,
                    item,
                    "character_field_not_text",
                    "character voice/look/name must be text; entry quarantined, nothing invented",
                    {"entry": bounded(info)},
                )
                continue
            if char_id not in merged:
                merged[char_id] = {"name": name, "bio": voice, "look": look}
            else:
                if voice and voice not in merged[char_id]["bio"]:
                    merged[char_id]["bio"] += " " + voice
                if look and look not in merged[char_id].get("look", ""):
                    merged[char_id]["look"] = (merged[char_id].get("look", "") + " " + look).strip()
    return merged


def _warn(ledger: RecoveryLedger | None, item: str, code: str, message: str, evidence: dict | None = None) -> None:
    if ledger is not None:
        ledger.record(STAGE, item, code, message, severity="quarantine", evidence=evidence)


# ##################################################################
# normalize for comparison
# strip accents and common prefixes for duplicate detection
def normalize_for_comparison(char_id: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", char_id)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    for prefix in ["the_", "don_", "dona_", "doña_"]:
        normalized = normalized.removeprefix(prefix)
    return normalized


# ##################################################################
# has obvious duplicates
# check if there are obvious duplicate patterns remaining
def has_obvious_duplicates(characters: dict) -> bool:
    char_ids = list(characters.keys())
    normalized_map = {}
    for char_id in char_ids:
        norm = normalize_for_comparison(char_id)
        if norm in normalized_map:
            return True
        normalized_map[norm] = char_id
    for i, id1 in enumerate(char_ids):
        for id2 in char_ids[i + 1 :]:
            if (id1 in id2 or id2 in id1) and id1 != "narrator" and id2 != "narrator":
                return True
    return False


# ##################################################################
# deduplicate characters
# use sonnet to identify and merge duplicate character entries in one call
async def deduplicate_characters(characters: dict, ledger: RecoveryLedger | None = None) -> dict:
    if len(characters) <= 1:
        return characters
    char_ids = list(characters.keys())
    char_summary = []
    for char_id in char_ids:
        info = characters[char_id]
        char_summary.append(f"{char_id}: {info['name']}")
    char_list = "\n".join(char_summary)
    prompt = f"""Deduplicate these character entries. Some refer to the SAME person under different names.

Character IDs and names:
{char_list}

MERGE THESE (same person, different names):
- Accent variations: "dona_sofia" = "doña_sofia"
- Title variations: "don_salvara" = "don_lorenzo_salvara" = "lorenzo"
- Article variations: "gray_king" = "the_gray_king"
- Name parts: "jean" = "jean_tannen", "calo" = "calo_sanza"
- Role names: "thiefmaker" = "the_thiefmaker"
- Character aliases/disguises should be merged with the real person

DO NOT MERGE (different people):
- Twins are DIFFERENT people (e.g., calo_sanza and galdo_sanza are separate)
- Sisters are DIFFERENT people (e.g., cheryn and raiza)
- Never create combined entries like "calo_and_galdo"

Return ONLY valid JSON:
{{
  "groups": [
    ["id1", "id2"],
    ["id3", "id4", "id5"]
  ]
}}

Each group = same person. IDs not in any group stay as singles."""

    response = (await ask(prompt)).strip()
    try:
        result = parse_json_response_strict(response)
        groups = result.get("groups", [])
        if not isinstance(groups, list):
            raise DataIssue("dedup_groups_not_list", "dedup 'groups' is not a list", {"response": bounded(response)})
    except (DataIssue, ValueError, TypeError, AttributeError) as error:
        # No merge is ever guessed: unmerged characters are kept and the issue is recorded.
        _warn(
            ledger,
            "deduplicate",
            getattr(error, "code", "dedup_response_malformed"),
            str(error),
            {"response": bounded(response)},
        )
        return characters
    return apply_dedup_groups(characters, groups, ledger)


# ##################################################################
# apply dedup groups
# merges only well-formed, unambiguous groups of known ids; malformed, unknown or overlapping groups are quarantined and never guessed at
def apply_dedup_groups(characters: dict, groups: list, ledger: RecoveryLedger | None = None) -> dict:
    if not groups:
        return characters
    id_to_canonical = {}
    claimed: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        if not isinstance(group, list) or not all(isinstance(g, str) for g in group):
            _warn(
                ledger,
                f"dedup-group-{group_index}",
                "dedup_group_malformed",
                "dedup group is not a list of text ids; skipped",
                {"group": bounded(group)},
            )
            continue
        valid_ids = [g for g in dict.fromkeys(group) if g in characters]
        unknown = [g for g in group if g not in characters]
        if unknown:
            _warn(
                ledger,
                f"dedup-group-{group_index}",
                "dedup_unknown_ids",
                "dedup group names ids that are not characters; they are ignored",
                {"unknown": [bounded(u, 80) for u in unknown]},
            )
        if len(valid_ids) < 2:
            continue
        if any(g in claimed for g in valid_ids):
            _warn(
                ledger,
                f"dedup-group-{group_index}",
                "dedup_group_ambiguous",
                "dedup group overlaps an earlier group (ambiguous identity); not merged",
                {"group": [bounded(g, 80) for g in valid_ids]},
            )
            continue
        for g in valid_ids:
            claimed[g] = group_index
        canonical = max(valid_ids, key=len)
        for char_id in valid_ids:
            id_to_canonical[char_id] = canonical
    deduplicated = {}
    for char_id, info in characters.items():
        canonical_id = id_to_canonical.get(char_id, char_id)
        if canonical_id not in deduplicated:
            deduplicated[canonical_id] = {"name": info["name"], "bio": info["bio"], "look": info.get("look", "")}
        else:
            existing_bio = deduplicated[canonical_id]["bio"]
            new_bio = info["bio"]
            if new_bio not in existing_bio:
                deduplicated[canonical_id]["bio"] += " " + new_bio
            new_look = info.get("look", "")
            if new_look and new_look not in deduplicated[canonical_id].get("look", ""):
                deduplicated[canonical_id]["look"] = (
                    deduplicated[canonical_id].get("look", "") + " " + new_look
                ).strip()
            if len(info["name"]) > len(deduplicated[canonical_id]["name"]):
                deduplicated[canonical_id]["name"] = info["name"]
    deduplicated = post_process_dedup(deduplicated, ledger)
    return deduplicated


# ##################################################################
# post process dedup
# fix common issues the llm misses
def post_process_dedup(characters: dict, ledger: RecoveryLedger | None = None) -> dict:
    # No book-specific or global alias table exists: identity merges come only from validated groups.
    result = dict(characters)
    invalid_merged = ["calo_and_galdo", "the_sanza_twins", "sanza_twins", "berangias_twins"]
    for invalid in invalid_merged:
        if invalid in result:
            _warn(
                ledger,
                f"combined-entry:{invalid}",
                "combined_character_entry_removed",
                "combined multi-person entry removed; individual entries are kept",
                {"entry": bounded(result[invalid])},
            )
            del result[invalid]
    return result


# ##################################################################
# create narrator entry
# generate narrator character based on book metadata and tone
async def create_narrator_entry(
    title: str, author: str, sample_text: str, ledger: RecoveryLedger | None = None
) -> dict:
    prompt = f"""Based on this book's title, author, and sample text, describe the ideal narrator.

Book: "{title}" by {author}

Sample text:
{sample_text[:3000]}

The narrator should have:
- Clarity and authority as a foundation
- A tone that matches the book's mood and genre
- Subtle personality influenced by what we know about the author or story

Return ONLY valid JSON:
{{
  "name": "Narrator",
  "bio": "A detailed description of the narrator's voice, tone, and personality for this specific book"
}}"""

    response = await query_haiku(prompt)
    try:
        entry = parse_json_response_strict(response)
        if not isinstance(entry.get("name"), str) or not isinstance(entry.get("bio"), str) or not entry["bio"].strip():
            raise DataIssue(
                "narrator_entry_malformed", "narrator entry lacks text name/bio", {"response": bounded(response)}
            )
        return {"name": entry["name"], "bio": entry["bio"]}
    except (DataIssue, ValueError, TypeError, AttributeError) as error:
        _warn(
            ledger,
            "narrator",
            getattr(error, "code", "narrator_entry_malformed"),
            str(error),
            {"response": bounded(response)},
        )
        return dict(DEFAULT_NARRATOR)


# ##################################################################
# analyze characters
# main entry point to analyze all chapters and produce characters.json
async def analyze_characters(output_dir: Path, title: str, author: str) -> Path:
    chapters_dir = output_dir / "chapters"
    characters_path = output_dir / "characters.json"
    if characters_path.exists():
        return characters_path
    chapter_files = sorted(chapters_dir.glob("*.txt"))
    if not chapter_files:
        raise OperationalError("no_chapters", f"no chapter files found in {chapters_dir}")
    sample_text = ""
    targets: list[tuple[int, Path]] = []
    for i, chapter_path in enumerate(chapter_files):
        if chapter_path.name == "00-intro.txt":
            continue
        if not sample_text:
            try:
                sample_text = chapter_path.read_text(encoding="utf-8")[:3000]
            except UnicodeDecodeError as error:
                RecoveryLedger(output_dir).record_error(
                    STAGE, chapter_path.name, error, severity="warning", evidence={"role": "narrator sample"}
                )
        targets.append((i, chapter_path))
    print(f"Analyzing {len(targets)} chapters in parallel...")
    ledger = RecoveryLedger(output_dir)

    async def one(index: int, path: Path) -> dict | None:
        # Only data problems quarantine the chapter; infrastructure errors propagate fail-closed.
        try:
            return await analyze_chapter(path, index)
        except Exception as error:
            if not is_data_error(error):
                raise
            ledger.record_error(STAGE, path.name, error, checkpoint={"chapter_index": index, "chapter": path.name})
            return None

    results = await asyncio.gather(*(one(i, p) for i, p in targets))
    all_chars = [r for r in results if r is not None]
    merged = merge_character_info(all_chars, ledger)
    print(f"Raw merge: {len(merged)} characters")
    print("Deduplicating with Sonnet...")
    deduplicated = await deduplicate_characters(merged, ledger)
    print(f"After dedup: {len(deduplicated)} characters")
    narrator_info = await create_narrator_entry(title, author, sample_text, ledger)
    deduplicated["narrator"] = narrator_info
    merged = deduplicated
    characters_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return characters_path


# ##################################################################
# analyze characters sync
# synchronous wrapper for analyze_characters
def analyze_characters_sync(output_dir: Path, title: str, author: str) -> Path:
    return asyncio.run(analyze_characters(output_dir, title, author))
