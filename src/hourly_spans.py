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


def parse_assignments(response: str, start: int, count: int, speakers: set[str]) -> dict[int, str]:
    seen: dict[int, str] = {}
    for raw in response.splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        if set(item) != {"index", "speaker_id"} or not isinstance(item["index"], int) or not isinstance(item["speaker_id"], str):
            raise ValueError("classifier output must contain only index and speaker_id")
        index, speaker = item["index"], item["speaker_id"]
        if index in seen or index < start or index >= start + count or speaker not in speakers:
            raise ValueError("classifier output has duplicate/out-of-range index or unknown speaker")
        seen[index] = speaker
    if set(seen) != set(range(start, start + count)):
        raise ValueError("classifier output is not a complete span bijection")
    return seen


async def classify_spans(text: str, speaker_ids: list[str]) -> list[dict]:
    spans = immutable_spans(text)
    speakers = set(speaker_ids)
    if "narrator" not in speakers:
        raise ValueError("speaker list needs narrator")
    assigned: dict[int, str] = {}
    for start in range(0, len(spans), BATCH_SIZE):
        batch = spans[start : start + BATCH_SIZE]
        indexed = "\n".join(f"{start+i}: {span}" for i, span in enumerate(batch))
        prompt = f"""Classify each immutable source span to exactly one audiobook speaker.
Valid speakers: {', '.join(speaker_ids)}. Return JSONL only: {{\"index\": number, \"speaker_id\": \"valid id\"}}.
Every listed index exactly once. narrator for narration and third-person prose. Direct speech may be quoted OR clearly attributed without quotes (for example, 'Klein said Look at it'); assign that speech to its named speaker when unambiguous. Never rewrite, copy, omit, or add text: the program constructs text locally from the immutable spans.

SPANS:\n{indexed}"""
        last = ""
        for attempt in range(6):
            try:
                response = await ask(prompt + (f"\nREPAIR: {last}" if last else ""))
                assigned.update(parse_assignments(response, start, len(batch), speakers))
                break
            except (ValueError, json.JSONDecodeError) as error:
                last = str(error)
        else:
            raise ValueError(f"span classifier failed batch {start}: {last}")
    return [{assigned[index]: span} for index, span in enumerate(spans)]


def generate_hourly_script_sync(chapter_path: Path, script_path: Path, speaker_ids: list[str]) -> Path:
    text = chapter_path.read_text(encoding="utf-8")
    lines = asyncio.run(classify_spans(text, speaker_ids))
    payload = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    temporary = script_path.with_suffix(".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(script_path)
    return script_path
