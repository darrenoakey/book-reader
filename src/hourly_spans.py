"""Immutable source-span classifier for hourly productions."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from src.llm import ask

SPAN_PATTERN = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.DOTALL)
BATCH_SIZE = 60


def immutable_spans(text: str) -> list[str]:
    spans = [match.group(0) for match in SPAN_PATTERN.finditer(text) if match.group(0).strip()]
    if not spans:
        raise ValueError("chapter has no immutable spans")
    return spans


def assignment_objects(response: str) -> list[object]:
    decoder = json.JSONDecoder()
    position = 0
    objects: list[object] = []
    while position < len(response):
        while position < len(response) and response[position].isspace():
            position += 1
        if position == len(response):
            break
        item, position = decoder.raw_decode(response, position)
        if isinstance(item, list):
            if objects or position != len(response.rstrip()):
                raise ValueError("classifier response mixes array with other JSON values")
            return item
        objects.append(item)
    return objects


# ##################################################################
# parse assignments
# accept valid JSON arrays or whitespace-delimited pretty JSON objects while enforcing a complete, exact assignment bijection.
def parse_assignments(response: str, start: int, count: int, speakers: set[str]) -> dict[int, str]:
    seen: dict[int, str] = {}
    for item in assignment_objects(response):
        if not isinstance(item, dict) or set(item) != {"index", "speaker_id"} or not isinstance(item["index"], int) or not isinstance(item["speaker_id"], str):
            raise ValueError("classifier output must contain only index and speaker_id")
        index, speaker = item["index"], item["speaker_id"]
        if index in seen or index < start or index >= start + count or speaker not in speakers:
            raise ValueError("classifier output has duplicate/out-of-range index or unknown speaker")
        seen[index] = speaker
    if set(seen) != set(range(start, start + count)):
        raise ValueError("classifier output is not a complete span bijection")
    return seen


def speaker_array_schema(speaker_ids: list[str], count: int) -> dict:
    return {
        "type": "array",
        "items": {"type": "string", "enum": speaker_ids},
        "minItems": count,
        "maxItems": count,
    }


# ##################################################################
# parse speakers
# bind each source-span position to exactly one schema-constrained speaker and reject malformed output without fallback.
def parse_speakers(response: str, count: int, speakers: set[str]) -> list[str]:
    values = json.loads(response)
    if not isinstance(values, list) or len(values) != count or any(not isinstance(value, str) or value not in speakers for value in values):
        raise ValueError("classifier schema response must be an exact list of valid speaker IDs")
    return values


async def classify_spans(text: str, speaker_ids: list[str], approved_aliases: dict[str, str] | None = None) -> list[dict]:
    spans = immutable_spans(text)
    speakers = set(speaker_ids)
    if "narrator" not in speakers:
        raise ValueError("speaker list needs narrator")
    assigned: dict[int, str] = {}
    for start in range(0, len(spans), BATCH_SIZE):
        batch = spans[start : start + BATCH_SIZE]
        indexed = "\n".join(f"{start+i}: {span}" for i, span in enumerate(batch))
        alias_notes = ", ".join(f"{alias}->{canonical}" for alias, canonical in sorted((approved_aliases or {}).items()) if alias != canonical)
        prompt = f"""Classify each immutable source span to exactly one audiobook speaker.
Valid speakers: {', '.join(speaker_ids)}. Approved aliases that must use their canonical speaker ID: {alias_notes or '(none)'}. Return only the JSON array specified by the response schema: one speaker ID per listed span, in exactly the listed order. narrator for narration and third-person prose. Direct speech may be quoted OR clearly attributed without quotes (for example, 'Klein said Look at it'); assign that speech to its named speaker when unambiguous. Never rewrite, copy, omit, or add text: the program constructs text locally from the immutable spans.

SPANS:\n{indexed}"""
        response = await ask(prompt, response_schema=speaker_array_schema(speaker_ids, len(batch)))
        try:
            assigned.update({start + index: speaker for index, speaker in enumerate(parse_speakers(response, len(batch), speakers))})
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"schema classifier failed batch {start}: {error}; response={response[:500]!r}") from error
    return [{assigned[index]: span} for index, span in enumerate(spans)]


def generate_hourly_script_sync(
    chapter_path: Path, script_path: Path, speaker_ids: list[str], approved_aliases: dict[str, str] | None = None
) -> Path:
    text = chapter_path.read_text(encoding="utf-8")
    lines = asyncio.run(classify_spans(text, speaker_ids, approved_aliases))
    payload = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    temporary = script_path.with_suffix(".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(script_path)
    return script_path
