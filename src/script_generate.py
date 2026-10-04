import asyncio
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

from src.data_recovery import OperationalError, RecoveryLedger, is_data_error
from src.hourly_spans import classify_spans
from src.llm import ask


# ##################################################################
# query haiku
# script generation via boringstack qwen3.6 (name kept for call-site
# compatibility — it is no longer Haiku)
async def query_haiku(prompt: str) -> str:
    return (await ask(prompt)).strip()


# ##################################################################
# parse jsonl response
# extract JSONL lines from a provider response while ignoring non-JSON lines
def parse_jsonl_response(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    result = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line and line.startswith("{"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "speaker_id" in entry and "text" in entry:
                entry = {entry["speaker_id"]: entry["text"]}
            elif "speaker" in entry and "text" in entry:
                entry = {entry["speaker"]: entry["text"]}
            result.append(entry)
    return result


# ##################################################################
# chunk text
# split chapter into pieces small enough that the model can handle reliably
def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 0) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ("\n\n", ". ", " "):
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap if end > overlap else end
    return chunks


# ##################################################################
# script integrity
# a script is only ever accepted whole: every chunk must parse completely,
# use known speakers and cover the source text. Anything else raises —
# there is NO all-narrator substitute.
CHUNK_SIZE = 4000
MAX_ATTEMPTS = 6
RETRY_DELAY_SECONDS = 5
SCRIPT_VERSION = 4
MIN_COVERAGE = 0.7
MAX_COVERAGE = 1.4
META_SUFFIX = ".meta.json"
_PROMPT_TEMPLATE = """Output JSONL only. No explanations. No markdown. Just JSONL lines.

Valid speakers: @@SPEAKERS@@

Convert to audiobook script. Each line MUST be exactly this JSON format:
{"speaker_id": "<one of the valid speakers>", "text": "<spoken words>"}

THE ONE HARD RULE: a character speaks ONLY text wrapped in quotation marks
(straight " " or ' ', or curly “ ” or ‘ ’). If text is NOT inside quotation
marks, it is NARRATION → "narrator". No exceptions, ever.

Therefore:
- If a sentence/passage contains NO quotation marks at all, EVERY line of it is "narrator".
- Third-person narration of a character's actions or thoughts is NARRATION, even
  when terse, clipped, or a subjectless fragment. "He vaulted the fence." → narrator.
  "Looked forward." → narrator. "He breathed. In. Out." → narrator. "Ducked." → narrator.
  A short fragment is NOT speech just because it's short — only quotation marks make it speech.
- The dialogue TAG ("Bob said", "she whispered", "he replied, grinning") is NARRATION
  → a separate "narrator" line. It is NOT part of the character's line.
- Split a sentence that mixes quoted speech and a tag into MULTIPLE lines.
- Do NOT include the quotation marks themselves in "text".
- speaker_id MUST be from the valid list. When unsure, use "narrator".
- One JSON object per line. No code fences, no explanation, no chapter title.

EXAMPLES:
Input: "We have to leave now," Bob said, glancing at the door.
Output:
{"speaker_id": "bob", "text": "We have to leave now,"}
{"speaker_id": "narrator", "text": "Bob said, glancing at the door."}

Input: The rain hammered the roof. "I won't," she snapped, "go back there."
Output:
{"speaker_id": "narrator", "text": "The rain hammered the roof."}
{"speaker_id": "jane", "text": "I won't,"}
{"speaker_id": "narrator", "text": "she snapped,"}
{"speaker_id": "jane", "text": "go back there."}

Input (clipped action prose, NO quotation marks — ALL narrator):
He didn't look back. Looked forward. Toward her. He vaulted a mailbox. Servos whined. He breathed. In. Out.
Output:
{"speaker_id": "narrator", "text": "He didn't look back. Looked forward. Toward her. He vaulted a mailbox. Servos whined. He breathed. In. Out."}

(Use the actual valid speaker_ids above, not "bob"/"jane", matching whoever is speaking.)

TEXT:
@@CHUNK@@

OUTPUT (JSONL only, nothing else):"""


class ScriptGenerationError(Exception):
    """Raised when a script cannot be produced or validated; never papered over."""


def build_prompt(chunk: str, speakers_list: str) -> str:
    return _PROMPT_TEMPLATE.replace("@@SPEAKERS@@", speakers_list).replace("@@CHUNK@@", chunk)


def _entry_pair(entry: object) -> tuple[str, str]:
    if not isinstance(entry, dict) or len(entry) != 1:
        raise ScriptGenerationError(f"malformed script entry: {entry!r}"[:200])
    ((speaker, text),) = entry.items()
    if not isinstance(speaker, str) or not isinstance(text, str):
        raise ScriptGenerationError(f"malformed script entry: {entry!r}"[:200])
    return speaker, text


def parse_jsonl_strict(text: str) -> list[dict]:
    """Parse a model response; any non-blank, non-fence line that is not a valid entry rejects the whole response."""
    result = []
    for raw in text.strip().split("\n"):
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScriptGenerationError(f"unparseable line: {line[:80]!r}") from exc
        if isinstance(entry, dict) and "text" in entry and ("speaker_id" in entry or "speaker" in entry):
            entry = {entry.get("speaker_id", entry.get("speaker")): entry["text"]}
        _entry_pair(entry)
        result.append(entry)
    if not result:
        raise ScriptGenerationError("empty response")
    return result


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum())


def validate_chunk(parsed: list[dict], chunk: str, speaker_ids: list[str]) -> None:
    allowed = set(speaker_ids)
    total = 0
    for entry in parsed:
        speaker, text = _entry_pair(entry)
        if speaker not in allowed:
            raise ScriptGenerationError(f"unknown speaker {speaker!r}")
        if not text.strip():
            raise ScriptGenerationError("empty text line")
        total += _letters(text)
    source = _letters(chunk)
    if source and not (MIN_COVERAGE * source <= total <= MAX_COVERAGE * source):
        raise ScriptGenerationError(f"coverage {total}/{source} letters outside accepted range")


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[\w]+", unicodedata.normalize("NFKC", text).casefold())


# ##################################################################
# validate script lines
# reject a script that loses, invents, or reorders source prose before it reaches audio.
def validate_script_lines(
    lines: list[dict], chapter_text: str, speaker_ids: list[str], include_title: bool = False
) -> None:
    body = lines if include_title else lines[1:]
    source_words = normalized_words(chapter_text)
    script_words: list[str] = []
    allowed = set(speaker_ids)
    if not source_words or not body:
        raise ScriptGenerationError("script has no source coverage")
    for entry in body:
        speaker, text = _entry_pair(entry)
        if speaker not in allowed or not text.strip():
            raise ScriptGenerationError("script has unknown speaker or empty text")
        script_words.extend(normalized_words(text))
    if script_words != source_words:
        mismatch = next(
            (i for i, pair in enumerate(zip(source_words, script_words)) if pair[0] != pair[1]),
            min(len(source_words), len(script_words)),
        )
        expected = source_words[mismatch] if mismatch < len(source_words) else "<end>"
        actual = script_words[mismatch] if mismatch < len(script_words) else "<end>"
        raise ScriptGenerationError(
            f"script source coverage differs at word {mismatch}: expected {expected!r}, got {actual!r}"
        )


# ##################################################################
# fingerprint + atomic cache
# canonical script = <script_dir>/<stem>.jsonl plus <stem>.jsonl.meta.json holding the
# input fingerprint and content hash. Consumers (hour runner) reuse via load_cached_script.
def script_fingerprint(
    chapter_text: str,
    chapter_title: str,
    speaker_ids: list[str],
    is_intro: bool = False,
) -> str:
    material = json.dumps(
        {
            "version": SCRIPT_VERSION,
            "prompt": hashlib.sha256(_PROMPT_TEMPLATE.encode()).hexdigest(),
            "chunk_size": CHUNK_SIZE,
            "text": hashlib.sha256(chapter_text.encode()).hexdigest(),
            "title": chapter_title,
            "intro": is_intro,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def meta_path_for(script_path: Path) -> Path:
    return script_path.with_name(script_path.name + META_SUFFIX)


def _read_lines(script_path: Path) -> list[dict] | None:
    try:
        return [json.loads(x) for x in script_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        return None


def load_cached_script(script_path: Path, fingerprint: str, allow_legacy: bool = False) -> list[dict] | None:
    """Return verified canonical lines; v2 cache remains immutable when explicitly accepted by a source validator."""
    try:
        meta = json.loads(meta_path_for(script_path).read_text(encoding="utf-8"))
        payload = script_path.read_bytes()
    except (OSError, ValueError):
        return None
    if meta.get("sha256") != hashlib.sha256(payload).hexdigest():
        return None
    if meta.get("fingerprint") != fingerprint and not (allow_legacy and meta.get("version") == 2):
        return None
    lines = _read_lines(script_path)
    return lines or None


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_canonical_script(script_path: Path, lines: list[dict], fingerprint: str) -> None:
    """Atomically publish script then meta; a script without matching meta is never treated as cached."""
    payload = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8")
    meta = {
        "fingerprint": fingerprint,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "version": SCRIPT_VERSION,
    }
    meta_path_for(script_path).unlink(missing_ok=True)
    _atomic_write(script_path, payload)
    _atomic_write(meta_path_for(script_path), json.dumps(meta).encode("utf-8"))


# ##################################################################
# generate chapter script
# convert chapter text to speaker-attributed jsonl — every chunk MUST succeed
async def _process_chunk(chunk: str, speaker_ids: list[str], label: str) -> list[dict]:
    speakers_list = ", ".join(speaker_ids)
    prompt = build_prompt(chunk, speakers_list)
    last_error = "no attempt"
    for attempt in range(MAX_ATTEMPTS):
        request = prompt
        if attempt:
            print(f"  {label} retry {attempt}: {last_error}")
            await asyncio.sleep(RETRY_DELAY_SECONDS)
            request += (
                "\nREPAIR REQUIRED: Your prior response failed exact source coverage: "
                f"{last_error}. Return every source word exactly once, in original order; "
                "only assign speakers and remove quotation marks. Output JSONL only."
            )
        response = await query_haiku(request)
        try:
            parsed = parse_jsonl_strict(response)
            validate_script_lines(parsed, chunk, speaker_ids, include_title=True)
            return parsed
        except ScriptGenerationError as exc:
            last_error = str(exc)
    raise ScriptGenerationError(f"{label}: no valid script after {MAX_ATTEMPTS} attempts ({last_error})")


async def generate_chapter_script(chapter_text: str, chapter_title: str, speaker_ids: list[str]) -> list[dict]:
    # The model chooses only a speaker ID. immutable_spans retains each source
    # sentence byte-for-byte, so a smaller backup model cannot omit or rewrite
    # narration while the existing canonical JSONL/cache contract is unchanged.
    lines = [{"narrator": chapter_title}, *await classify_spans(chapter_text, speaker_ids)]
    validate_script_lines(lines, chapter_text, speaker_ids)
    return lines


# ##################################################################
# generate script for chapter file
# process a single chapter file to jsonl
async def generate_script_for_file(chapter_path: Path, script_dir: Path, speaker_ids: list[str]) -> Path:
    script_name = chapter_path.stem + ".jsonl"
    script_path = script_dir / script_name
    chapter_text = chapter_path.read_text(encoding="utf-8")
    chapter_title = chapter_path.stem.split("-", 1)[-1].replace("_", " ").title()
    is_intro = chapter_path.name == "00-intro.txt"
    fingerprint = script_fingerprint(chapter_text, chapter_title, speaker_ids, is_intro)
    cached = load_cached_script(script_path, fingerprint, allow_legacy=True)
    if cached is not None:
        if is_intro:
            return script_path
        try:
            validate_script_lines(cached, chapter_text, speaker_ids)
            return script_path
        except ScriptGenerationError:
            pass
    if is_intro:
        lines = [{"narrator": chapter_text}]
    else:
        lines = await generate_chapter_script(chapter_text, chapter_title, speaker_ids)
        validate_script_lines(lines, chapter_text, speaker_ids)
    await asyncio.to_thread(write_canonical_script, script_path, lines, fingerprint)
    return script_path


# ##################################################################
# get speaker ids
# load speaker ids from voices.json
def get_speaker_ids(output_dir: Path) -> list[str]:
    voices_path = output_dir / "voices.json"
    if not voices_path.exists():
        raise OperationalError("voices_missing", "voices.json not found")
    try:
        voices = json.loads(voices_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OperationalError("voices_unreadable", f"voices.json is unreadable: {voices_path}") from error
    if not isinstance(voices, dict) or not voices:
        raise OperationalError("voices_invalid", f"voices.json is not a non-empty object: {voices_path}")
    return list(voices.keys())


# ##################################################################
# generate single script
# process just one chapter by number
async def generate_single_script(output_dir: Path, chapter_num: int) -> Path:
    speaker_ids = get_speaker_ids(output_dir)
    chapters_dir = output_dir / "chapters"
    script_dir = output_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    chapter_files = sorted(chapters_dir.glob("*.txt"))
    if chapter_num < 0 or chapter_num >= len(chapter_files):
        raise ValueError(f"Chapter {chapter_num} not found (have {len(chapter_files)} chapters)")
    chapter_path = chapter_files[chapter_num]
    script_name = chapter_path.stem + ".jsonl"
    script_path = script_dir / script_name
    script_path.unlink(missing_ok=True)
    meta_path_for(script_path).unlink(missing_ok=True)
    return await generate_script_for_file(chapter_path, script_dir, speaker_ids)


# ##################################################################
# generate single script sync
# synchronous wrapper for generate_single_script
def generate_single_script_sync(output_dir: Path, chapter_num: int) -> Path:
    return asyncio.run(generate_single_script(output_dir, chapter_num))


# ##################################################################
# generate all scripts
# process all chapters to jsonl scripts
async def generate_all_scripts(output_dir: Path) -> list[Path]:
    speaker_ids = get_speaker_ids(output_dir)
    chapters_dir = output_dir / "chapters"
    script_dir = output_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    chapter_files = sorted(chapters_dir.glob("*.txt"))
    print(f"Generating {len(chapter_files)} chapter scripts in parallel...")
    ledger = RecoveryLedger(output_dir)

    async def one(index: int, path: Path) -> Path | None:
        # A data problem quarantines this chapter only (no script is published for it, nothing is
        # fabricated); infrastructure/store failures propagate fail-closed.
        try:
            return await generate_script_for_file(path, script_dir, speaker_ids)
        except Exception as error:
            if not is_data_error(error) and not isinstance(error, ScriptGenerationError):
                raise
            ledger.record_error("scripts", path.name, error, checkpoint={"chapter_index": index, "chapter": path.name})
            return None

    results = await asyncio.gather(*(one(i, p) for i, p in enumerate(chapter_files)))
    return [path for path in results if path is not None]


# ##################################################################
# generate scripts sync
# synchronous wrapper for generate_all_scripts
def generate_scripts_sync(output_dir: Path) -> list[Path]:
    return asyncio.run(generate_all_scripts(output_dir))
