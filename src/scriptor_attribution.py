"""Speaker attribution with the fine-tuned scriptor model (Qwen3-0.6B LoRA, trained in ~/src/scriptor).

The model reads a passage plus the previous script lines and a speaker list, and returns JSONL
{"speaker_id": "words"}. It decides only WHERE lines break and WHO speaks them: every returned
script line is a raw slice of the chapter, so "".join(lines) == chapter byte-for-byte whatever the
model emits. The prompt format below must match scriptor's training format exactly
(scriptor/doc.py SYSTEM_PROMPT, user_message, speaker_listing, clip_previous).
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from src.llm import Backend, RouterConfig, ask_sync

NARRATOR = "narrator"
UNKNOWN = "unknown"
SYSTEM_PROMPT = (
    "You turn book prose into an audiobook script. Output one JSON object per line: "
    '{"<speaker_id>": "<text>"}. Use "narrator" for all narration, including dialogue tags, '
    "action beats, thoughts and reported speech. A character's line holds only the words that "
    "character says aloud, without the quotation marks. Use a speaker id from the list, or "
    '"unknown" for a speaker who is not listed. Start a new line at every change of speaker and '
    "at every paragraph break. Copy every word of the passage exactly once, in order."
)
WORD = re.compile(r"\w+")
QUOTES = "\"\u201c\u201d\u2018\u2019'"
SENTENCE_END = re.compile(r"[.!?\u2026][\"\u201d\u2019']?\s+")
PASSAGE_CHARS = 2200
PREVIOUS_LINES = 10
MAX_LISTED = 20
RECENT_SPEAKER_LINES = 40
MAX_TOKENS = 3000

DEFAULT_MODEL = "scriptor:v1-1000"
PRIMARY_URL = "http://10.0.0.42:11434"
BACKUP_URL = "http://127.0.0.1:11434"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "local" / "config.toml"

Chat = Callable[[list[dict]], str]


@dataclass(frozen=True)
class Character:
    id: str
    name: str
    gender: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


# ##################################################################
# scriptor router config
# the small model is served by native Ollama on the boringstack (primary) and this Mac (backup); the
# same ICMP-gated routing as every other book-reader call, overridable in local/config.toml [scriptor].
def load_scriptor_config(path: Path = CONFIG_PATH) -> RouterConfig:
    values: dict = {}
    if path.is_file():
        with path.open("rb") as stream:
            values = tomllib.load(stream).get("scriptor", {})
    model = str(values.get("model", DEFAULT_MODEL))

    def backend(name: str, url: str, host: str) -> Backend:
        identity = values.get(f"{name}_identity")
        return Backend(
            str(values.get(f"{name}_url", url)).rstrip("/"),
            str(values.get(f"{name}_ping_host", host)),
            str(values.get(f"{name}_model", model)),
            "ollama",
            int(values.get("num_ctx", 8192)),
            False,
            str(identity) if identity else None,
        )

    return RouterConfig(backend("primary", PRIMARY_URL, "10.0.0.42"), backend("backup", BACKUP_URL, "127.0.0.1"), 1)


def scriptor_chat(config: RouterConfig) -> Chat:
    def chat(messages: list[dict]) -> str:
        system, user = messages[0]["content"], messages[1]["content"]
        return ask_sync(user, system=system, temperature=0.0, max_tokens=MAX_TOKENS, config=config)

    return chat


# ##################################################################
# prompt (identical to training)
def speaker_listing(characters: list[Character]) -> str:
    rows = []
    for char in characters:
        names = [char.name] + [a for a in char.aliases if a and a != char.name]
        gender = f" [{char.gender}]" if char.gender in ("f", "m", "n") else ""
        rows.append(f"{char.id}{gender}: {' / '.join(dict.fromkeys(names))}")
    return "\n".join(rows) if rows else "(none)"


def format_line(speaker: str, text: str) -> str:
    return json.dumps({speaker: text}, ensure_ascii=False)


def user_message(characters: list[Character], previous: list[tuple[str, str]], passage: str) -> str:
    prev = "\n".join(format_line(s, t) for s, t in previous) if previous else "(start of text)"
    return f"Speakers:\n{speaker_listing(characters)}\n\nPrevious lines:\n{prev}\n\nPassage:\n{passage}"


def clip_previous(lines: list[tuple[str, str]], limit: int = 400) -> list[tuple[str, str]]:
    out = []
    for speaker, text in lines:
        if len(text) > limit:
            cut = text[-limit:]
            cut = cut[cut.find(" ") + 1 :] if " " in cut else cut
            text = "\u2026" + cut
        out.append((speaker, text))
    return out


# ##################################################################
# parse model output
# one {"speaker": "text"} per line; unlisted speakers become unknown, junk lines are skipped.
def parse_lines(text: str, allowed: set[str]) -> list[tuple[str, str]]:
    out = []
    for raw in text.strip().splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or len(obj) != 1:
            continue
        ((speaker, value),) = obj.items()
        if isinstance(speaker, str) and isinstance(value, str):
            out.append((speaker if speaker in allowed else UNKNOWN, value))
    return out


def _norm(word: str) -> str:
    return unicodedata.normalize("NFKC", word).casefold()


# ##################################################################
# align
# each source word is owned by the predicted line whose word it matched; unmatched words join the
# preceding line; ownership is monotonic so lines never reorder.
def _owners(passage: str, predicted: list[tuple[str, str]]) -> tuple[list[tuple[int, int, str]], list[int]]:
    src = [(m.start(), m.end(), _norm(m.group())) for m in WORD.finditer(passage)]
    pred = [(_norm(w), index) for index, (_, text) in enumerate(predicted) for w in WORD.findall(text)]
    owner: list[int | None] = [None] * len(src)
    matcher = difflib.SequenceMatcher(a=[w for *_, w in src], b=[w for w, _ in pred], autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            owner[block.a + k] = pred[block.b + k][1]
    last = next((o for o in owner if o is not None), None)
    if last is None:
        return src, []
    filled: list[int] = []
    for o in owner:
        last = last if o is None else o
        filled.append(max(last, filled[-1]) if filled else last)
    return src, filled


def _cut(passage: str, prev_end: int, next_start: int) -> int:
    """Where the left line ends between two words: closing quotes/punctuation stay left, the rest leads right."""
    gap = passage[prev_end:next_start]
    if "\n" in gap:
        return prev_end + gap.index("\n")
    ws = re.search(r"\s", gap)
    if ws is None:
        quote = re.search(r"[\u201c\u2018\"]", gap)
        return prev_end + (quote.start() if quote else len(gap))
    return prev_end + ws.start()


def partition_passage(passage: str, predicted: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Tile the raw passage into (speaker, slice) pairs whose slices concatenate to the passage exactly."""
    src, owner = _owners(passage, predicted)
    if not src or not owner:
        return [(NARRATOR, passage)] if passage else []
    starts = [0] + [
        i for i in range(1, len(owner)) if owner[i] != owner[i - 1] or "\n" in passage[src[i - 1][1] : src[i][0]]
    ]
    bounds = [0] + [_cut(passage, src[i - 1][1], src[i][0]) for i in starts[1:]] + [len(passage)]
    out = []
    for n, first in enumerate(starts):
        piece = passage[bounds[n] : bounds[n + 1]]
        if piece:
            out.append((predicted[owner[first]][0], piece))
    return out


# ##################################################################
# passages
# tile the chapter into model-sized passages cut at paragraph breaks (or sentence ends inside huge paragraphs).
def passage_bounds(text: str, limit: int = PASSAGE_CHARS) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\n]*\S[^\n]*", text):
        start, end = match.start(), match.end()
        while end - start > limit:
            window = text[start : start + limit]
            cuts = [m.start() + len(m.group().rstrip()) for m in SENTENCE_END.finditer(window)]
            cuts = [c for c in cuts if c > limit // 3]
            if not cuts:
                space = window.rfind(" ", limit // 3)
                cuts = [space] if space > 0 else [limit]
            pieces.append((start, start + cuts[-1]))
            start += cuts[-1]
        pieces.append((start, end))
    if not pieces:
        return [(0, len(text))] if text else []
    groups: list[list[int]] = []
    for start, end in pieces:
        if groups and end - groups[-1][0] <= limit:
            groups[-1][1] = end
        else:
            groups.append([start, end])
    bounds = [0] + [groups[k][1] for k in range(len(groups) - 1)] + [len(text)]
    return [(bounds[k], bounds[k + 1]) for k in range(len(groups))]


def _mentions(text: str, char: Character) -> bool:
    lowered = text.casefold()
    return any(
        len(name.strip()) >= 2 and re.search(rf"(?<!\w){re.escape(name.strip().casefold())}(?!\w)", lowered)
        for name in (char.name, *char.aliases)
    )


# ##################################################################
# speaker list
# the model was trained on short lists (passage speakers plus a few others); a whole-book cast is
# narrowed to names in the passage, recent speakers, then names in the recent context.
def select_characters(
    characters: list[Character], passage: str, history: list[tuple[str, str]], limit: int = MAX_LISTED
) -> list[Character]:
    by_id = {c.id: c for c in characters}
    recent: list[str] = []
    for speaker, _ in reversed(history[-RECENT_SPEAKER_LINES:]):
        if speaker in by_id and speaker not in recent:
            recent.append(speaker)
    context = passage + "\n" + "\n".join(t for _, t in history[-PREVIOUS_LINES:])
    in_passage = [c.id for c in characters if _mentions(passage, c)]
    named = [c.id for c in characters if _mentions(context, c)]
    ordered = dict.fromkeys([*[i for i in recent if i in in_passage], *in_passage, *recent, *named])
    return [by_id[i] for i in list(ordered)[:limit]]


def _history_text(speaker: str, piece: str) -> str:
    text = " ".join(piece.split())
    return text.strip(QUOTES + " ").strip() or text if speaker != NARRATOR else text


# ##################################################################
# partition chapter
# run the model passage by passage, feeding back the previous script lines, and return exact source slices.
def partition_chapter(
    text: str, characters: list[Character], chat: Chat, history: list[tuple[str, str]] | None = None
) -> list[tuple[str, str]]:
    allowed = {NARRATOR, UNKNOWN} | {c.id for c in characters}
    history = list(history or [])
    result: list[tuple[str, str]] = []
    for start, end in passage_bounds(text):
        raw = text[start:end]
        if not WORD.search(raw):
            result.append((NARRATOR, raw))
            continue
        listed = select_characters(characters, raw, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(listed, clip_previous(history[-PREVIOUS_LINES:]), raw.strip())},
        ]
        pieces = partition_passage(raw, parse_lines(chat(messages), allowed))
        result.extend(pieces)
        history.extend((s, _history_text(s, p)) for s, p in pieces if WORD.search(p))
    return result


def characters_for(
    speaker_ids: list[str], names: dict[str, str] | None = None, aliases: dict[str, str] | None = None
) -> list[Character]:
    """Speaker ids (+ display names and approved alias ids) -> the model's speaker list; narrator is implicit."""
    extra: dict[str, list[str]] = {}
    for alias, canonical in (aliases or {}).items():
        if alias != canonical:
            extra.setdefault(canonical, []).append(alias.replace("_", " ").title())
    out = []
    for sid in speaker_ids:
        if sid == NARRATOR:
            continue
        name = ((names or {}).get(sid) or sid.replace("_", " ").title()).strip()
        out.append(Character(sid, name, "", tuple(extra.get(sid, []))))
    return out
