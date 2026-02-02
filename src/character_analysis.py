import asyncio
import json
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query


# ##################################################################
# parse json response
# extract json from claude response handling markdown code blocks and preamble
def parse_json_response(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"characters": {}}
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            block = text[start:end + 3]
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
                text = text[:end_brace + 1]
    if not text or not text.startswith("{"):
        return {"characters": {}}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"characters": {}}


# ##################################################################
# query haiku
# send a prompt to claude haiku and get text response
async def query_haiku(prompt: str) -> str:
    response = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=[],
            permission_mode="bypassPermissions",
            model="haiku",
        )
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response += block.text
    return response.strip()


# ##################################################################
# analyze chapter
# extract character information from a single chapter
async def analyze_chapter(chapter_path: Path, chapter_num: int) -> dict:
    text = chapter_path.read_text(encoding="utf-8")
    prompt = f"""Analyze this chapter and identify characters who speak or have internal monologue.

For each speaking character, extract ONLY voice-relevant physical details:
- Gender (from pronouns or descriptions)
- Age (child, young adult, middle-aged, elderly, or specific if mentioned)
- Physical build (large, small, thin, heavy, etc.)
- Ethnicity or accent hints (nationality, regional background)
- Voice/speech patterns (gruff, soft, educated, crude, accent, lisping, etc.)
- Distinctive physical traits affecting voice (old, frail, booming, wheezing, etc.)

Return ONLY valid JSON:
{{
  "characters": {{
    "character_id": {{
      "name": "Display Name",
      "details": "Physical and voice description only"
    }}
  }}
}}

Rules:
- Include ONLY characters who actually speak (quoted dialogue) or have internal monologue
- Do NOT include characters who are merely mentioned
- Character IDs: lowercase with underscores (e.g., "jean_tannen")
- Details must focus on VOICE generation - what would help create their voice
- EXCLUDE: plot roles, story function, relationships to other characters, emotional descriptions
- INCLUDE: "elderly man with gravelly voice" / "young woman, speaks formally" / "large man, booming voice"
- NO cross-character references (don't mention other characters in the details)

Chapter {chapter_num} text:
{text[:15000]}"""

    response = await query_haiku(prompt)
    return parse_json_response(response)


# ##################################################################
# merge character info
# combine character info from multiple chapters
def merge_character_info(all_chars: list[dict]) -> dict:
    merged = {}
    for chapter_chars in all_chars:
        for char_id, info in chapter_chars.get("characters", {}).items():
            details = info.get("details", info.get("bio", ""))
            name = info.get("name", char_id)
            if char_id not in merged:
                merged[char_id] = {"name": name, "bio": details}
            else:
                merged[char_id]["bio"] += " " + details
    return merged


# ##################################################################
# normalize for comparison
# strip accents and common prefixes for duplicate detection
def normalize_for_comparison(char_id: str) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFKD", char_id)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    for prefix in ["the_", "don_", "dona_", "doña_"]:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
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
        for id2 in char_ids[i + 1:]:
            if id1 in id2 or id2 in id1:
                if id1 != "narrator" and id2 != "narrator":
                    return True
    return False


# ##################################################################
# deduplicate characters
# use sonnet to identify and merge duplicate character entries in one call
async def deduplicate_characters(characters: dict) -> dict:
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

    response = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=[],
            permission_mode="bypassPermissions",
            model="sonnet",
        )
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response += block.text
    response = response.strip()
    result = parse_json_response(response)
    groups = result.get("groups", [])
    if not groups:
        return characters
    id_to_canonical = {}
    for group in groups:
        if not group:
            continue
        valid_ids = [g for g in group if g in characters]
        if not valid_ids:
            continue
        canonical = max(valid_ids, key=len)
        for char_id in group:
            if char_id in characters:
                id_to_canonical[char_id] = canonical
    deduplicated = {}
    for char_id, info in characters.items():
        canonical_id = id_to_canonical.get(char_id, char_id)
        if canonical_id not in deduplicated:
            deduplicated[canonical_id] = {
                "name": info["name"],
                "bio": info["bio"]
            }
        else:
            existing_bio = deduplicated[canonical_id]["bio"]
            new_bio = info["bio"]
            if new_bio not in existing_bio:
                deduplicated[canonical_id]["bio"] += " " + new_bio
            if len(info["name"]) > len(deduplicated[canonical_id]["name"]):
                deduplicated[canonical_id]["name"] = info["name"]
    deduplicated = post_process_dedup(deduplicated)
    return deduplicated


# ##################################################################
# post process dedup
# fix common issues the llm misses
def post_process_dedup(characters: dict) -> dict:
    result = dict(characters)
    known_aliases = {
        "tavrin_callas": "jean_tannen",
        "lukas_fehrwight": "locke_lamora",
        "capa_raza": "the_gray_king",
    }
    for alias, canonical in known_aliases.items():
        if alias in result and canonical in result:
            result[canonical]["bio"] += " " + result[alias]["bio"]
            del result[alias]
    invalid_merged = ["calo_and_galdo", "the_sanza_twins", "sanza_twins", "berangias_twins"]
    for invalid in invalid_merged:
        if invalid in result:
            del result[invalid]
    return result


# ##################################################################
# create narrator entry
# generate narrator character based on book metadata and tone
async def create_narrator_entry(title: str, author: str, sample_text: str) -> dict:
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
    return parse_json_response(response)


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
        raise ValueError("No chapter files found")
    all_chars = []
    sample_text = ""
    for i, chapter_path in enumerate(chapter_files):
        if chapter_path.name == "00-intro.txt":
            continue
        if not sample_text:
            sample_text = chapter_path.read_text(encoding="utf-8")[:3000]
        chapter_chars = await analyze_chapter(chapter_path, i)
        all_chars.append(chapter_chars)
    merged = merge_character_info(all_chars)
    print(f"Raw merge: {len(merged)} characters")
    print("Deduplicating with Sonnet...")
    deduplicated = await deduplicate_characters(merged)
    print(f"After dedup: {len(deduplicated)} characters")
    narrator_info = await create_narrator_entry(title, author, sample_text)
    deduplicated["narrator"] = narrator_info
    merged = deduplicated
    characters_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return characters_path


# ##################################################################
# analyze characters sync
# synchronous wrapper for analyze_characters
def analyze_characters_sync(output_dir: Path, title: str, author: str) -> Path:
    return asyncio.run(analyze_characters(output_dir, title, author))
