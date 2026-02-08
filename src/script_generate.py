import asyncio
import json
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query


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
            result.append(entry)
    return result


# ##################################################################
# generate chapter script
# convert chapter text to speaker-attributed jsonl
async def generate_chapter_script(chapter_text: str, chapter_title: str, speaker_ids: list[str]) -> list[dict]:
    speakers_list = ", ".join(speaker_ids)
    prompt = f"""Output JSONL only. No explanations. No markdown. Just JSONL lines.

Valid speakers: {speakers_list}

Convert to audiobook script:
- Quoted dialogue → character's speaker_id
- Everything else → narrator
- One speaker per line

Start with: {{"narrator": "{chapter_title}"}}

TEXT:
{chapter_text[:12000]}

OUTPUT (JSONL only, nothing else):"""

    response = await query_haiku(prompt)
    return parse_jsonl_response(response)


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
    # Delete existing script to regenerate
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
    created = []
    for chapter_path in chapter_files:
        script_path = await generate_script_for_file(chapter_path, script_dir, speaker_ids)
        created.append(script_path)
    return created


# ##################################################################
# generate scripts sync
# synchronous wrapper for generate_all_scripts
def generate_scripts_sync(output_dir: Path) -> list[Path]:
    return asyncio.run(generate_all_scripts(output_dir))
