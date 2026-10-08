"""Immutable source-span classifier for hourly productions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from src.scriptor_attribution import characters_for, load_scriptor_config, partition_chapter, scriptor_chat

SPAN_PATTERN = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.DOTALL)


def immutable_spans(text: str) -> list[str]:
    spans = [match.group(0) for match in SPAN_PATTERN.finditer(text) if match.group(0).strip()]
    if not spans:
        raise ValueError("chapter has no immutable spans")
    return spans


# ##################################################################
# scoped reference notes
# turns exact validated mention references into per-span prompt notes keyed by this chapter's content hash and each span's hash and offset; a reference never applies to any other mention of the same name.
def scoped_reference_notes(
    text: str, spans: list[str], speakers: set[str], references: list[dict] | None
) -> dict[int, list[str]]:
    chapter_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    by_span: dict[str, list[dict]] = {}
    for reference in references or []:
        if reference["chapter_sha256"] == chapter_hash:
            by_span.setdefault(reference["quote_sha256"], []).append(reference)
    notes: dict[int, list[str]] = {}
    for index, span in enumerate(spans):
        for reference in sorted(
            by_span.get(hashlib.sha256(span.encode("utf-8")).hexdigest(), []), key=lambda r: r["span_start"]
        ):
            label, start = reference["label"], reference["span_start"]
            if span[start : start + len(label)] != label:
                raise ValueError("scoped reference does not match its source span")
            if reference["canonical"] not in speakers:
                raise ValueError(
                    f"scoped reference targets a speaker outside the valid speakers: {reference['canonical']}"
                )
            notes.setdefault(index, []).append(
                f"span {index}: the mention {label!r} at character offset {start} is the character {reference['canonical']}"
            )
    return notes


# ##################################################################
# classify spans
# the fine-tuned scriptor model splits quotes from dialogue tags and names each speaker; every returned
# line is an exact source slice, so "".join(lines) == text. Scoped references are validated (fail closed
# on a stale or mismatched reference) but the small model takes names only from the speaker list.
async def classify_spans(
    text: str,
    speaker_ids: list[str],
    approved_aliases: dict[str, str] | None = None,
    scoped_references: list[dict] | None = None,
    names: dict[str, str] | None = None,
) -> list[dict]:
    spans = immutable_spans(text)
    speakers = set(speaker_ids)
    if "narrator" not in speakers:
        raise ValueError("speaker list needs narrator")
    scoped_reference_notes(text, spans, speakers, scoped_references)
    characters = characters_for(sorted(speakers), names, approved_aliases)
    chat = scriptor_chat(load_scriptor_config())
    lines = await asyncio.to_thread(partition_chapter, text, characters, chat)
    script = [{speaker if speaker in speakers else "narrator": piece} for speaker, piece in lines]
    if "".join(next(iter(line.values())) for line in script) != text:
        raise ValueError("scriptor attribution did not reconstruct the chapter exactly")
    return script


def generate_hourly_script_sync(
    chapter_path: Path,
    script_path: Path,
    speaker_ids: list[str],
    approved_aliases: dict[str, str] | None = None,
    scoped_references: list[dict] | None = None,
    names: dict[str, str] | None = None,
) -> Path:
    text = chapter_path.read_text(encoding="utf-8")
    lines = asyncio.run(classify_spans(text, speaker_ids, approved_aliases, scoped_references, names))
    payload = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    temporary = script_path.with_suffix(".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(script_path)
    return script_path
