import asyncio
import json
from pathlib import Path

from src.llm import ask


# ##################################################################
# query haiku
# script generation via boringstack qwen3.6 (name kept for call-site
# compatibility — it is no longer Haiku)
async def query_haiku(prompt: str) -> str:
    return (await ask(prompt)).strip()


# ##################################################################
# parse jsonl response
# extract jsonl lines from claude response skipping non-json lines
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
# generate chapter script
# convert chapter text to speaker-attributed jsonl — every chunk MUST succeed
async def _process_chunk(chunk: str, speakers_list: str, label: str) -> list[dict]:
    prompt = f"""Output JSONL only. No explanations. No markdown. Just JSONL lines.

Valid speakers: {speakers_list}

Convert to audiobook script. Each line MUST be exactly this JSON format:
{{"speaker_id": "<one of the valid speakers>", "text": "<spoken words>"}}

Rules:
- ONLY the words INSIDE quotation marks are a character's speech → that character's speaker_id.
- The dialogue TAG ("Bob said", "she whispered", "he replied, grinning") is NARRATION → a separate "narrator" line. It is NOT part of the character's line.
- All other prose (description, action, narration) → "narrator".
- Split a sentence that mixes speech and tag into MULTIPLE lines.
- Do NOT include the quotation marks themselves in "text".
- speaker_id MUST be from the valid list. If unsure, use "narrator".
- One JSON object per line. No code fences, no explanation, no chapter title.

EXAMPLES:
Input: "We have to leave now," Bob said, glancing at the door.
Output:
{{"speaker_id": "bob", "text": "We have to leave now,"}}
{{"speaker_id": "narrator", "text": "Bob said, glancing at the door."}}

Input: The rain hammered the roof. "I won't," she snapped, "go back there."
Output:
{{"speaker_id": "narrator", "text": "The rain hammered the roof."}}
{{"speaker_id": "jane", "text": "I won't,"}}
{{"speaker_id": "narrator", "text": "she snapped,"}}
{{"speaker_id": "jane", "text": "go back there."}}

(Use the actual valid speaker_ids above, not "bob"/"jane", matching whoever is speaking.)

TEXT:
{chunk}

OUTPUT (JSONL only, nothing else):"""
    response = await query_haiku(prompt)
    parsed = parse_jsonl_response(response)
    attempt = 0
    while not parsed and attempt < 5:
        attempt += 1
        response = await query_haiku(prompt)
        parsed = parse_jsonl_response(response)
        if not parsed:
            print(f"  {label} parse retry {attempt}")
            await asyncio.sleep(5)
    if not parsed:
        # All-narrator fallback for chunks the model can't structure (e.g.
        # pure narration with no dialogue keeps producing unparseable output).
        print(f"  {label} giving up after {attempt} retries — narrator fallback")
        parsed = [{"narrator": chunk}]
    return parsed


async def generate_chapter_script(chapter_text: str, chapter_title: str, speaker_ids: list[str]) -> list[dict]:
    speakers_list = ", ".join(speaker_ids)
    chunks = chunk_text(chapter_text, chunk_size=4000)
    chunk_results = await asyncio.gather(
        *(_process_chunk(c, speakers_list, f"{chapter_title} chunk {i+1}/{len(chunks)}")
          for i, c in enumerate(chunks))
    )
    all_lines: list[dict] = [{"narrator": chapter_title}]
    for parsed in chunk_results:
        all_lines.extend(parsed)
    return all_lines


# ##################################################################
# generate script for chapter file
# process a single chapter file to jsonl
async def generate_script_for_file(chapter_path: Path, script_dir: Path, speaker_ids: list[str]) -> Path:
    script_name = chapter_path.stem + ".jsonl"
    script_path = script_dir / script_name
    if script_path.exists():
        return script_path
    chapter_text = chapter_path.read_text(encoding="utf-8")
    chapter_title = chapter_path.stem.split("-", 1)[-1].replace("_", " ").title()
    if chapter_path.name == "00-intro.txt":
        lines = [{"narrator": chapter_text}]
    else:
        lines = await generate_chapter_script(chapter_text, chapter_title, speaker_ids)
    with open(script_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return script_path


# ##################################################################
# get speaker ids
# load speaker ids from voices.json
def get_speaker_ids(output_dir: Path) -> list[str]:
    voices_path = output_dir / "voices.json"
    if not voices_path.exists():
        raise ValueError("voices.json not found")
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
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
    if script_path.exists():
        script_path.unlink()
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
    return list(await asyncio.gather(
        *(generate_script_for_file(p, script_dir, speaker_ids) for p in chapter_files)
    ))


# ##################################################################
# generate scripts sync
# synchronous wrapper for generate_all_scripts
def generate_scripts_sync(output_dir: Path) -> list[Path]:
    return asyncio.run(generate_all_scripts(output_dir))
