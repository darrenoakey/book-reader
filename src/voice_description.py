import asyncio
import json
from pathlib import Path

from src.llm import ask


# ##################################################################
# query sonnet
# voice-description generation via boringstack qwen3.6 (kept the name for
# call-site compatibility — it is no longer Sonnet)
async def query_sonnet(prompt: str) -> str:
    return (await ask(prompt)).strip()


# ##################################################################
# parse json response
# extract json from claude response handling markdown code blocks
def parse_json_response(text: str) -> dict:
    text = text.strip()
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
    return json.loads(text)


async def _voice_description_for_one(char_id: str, info: dict) -> tuple[str, dict]:
    bio = info.get("bio") or info.get("description") or ""
    is_narrator = char_id == "narrator"
    role_hint = ("Audiobook narrator. Focus on narration qualities: clarity, authority, "
                 "warmth, pace, range for character voices.") if is_narrator else \
                "Audiobook character voice for TTS cloning."
    prompt = f"""You are designing a voice for an audiobook character. Output a concise (60-100 word) voice description suitable for TTS voice cloning.

{role_hint}

Character id: {char_id}
Bio: {bio}

Cover: gender, age, accent/dialect, pitch, timbre, pace, emotion, distinctive vocal traits. Be specific and vivid.

Output the voice description text only. No JSON, no preamble."""
    while True:
        text = (await query_sonnet(prompt)).strip()
        if text:
            return char_id, {"description": text}
        print(f"  empty response for {char_id} — retrying")
        await asyncio.sleep(3)


# ##################################################################
# generate all voice descriptions
# per-character parallel sonnet calls (batched JSON triggers refusal)
async def generate_all_voice_descriptions(characters: dict) -> dict:
    print(f"  voice descriptions: {len(characters)} parallel sonnet calls")
    results = await asyncio.gather(
        *(_voice_description_for_one(cid, info) for cid, info in characters.items())
    )
    return {cid: desc for cid, desc in results}


# ##################################################################
# generate voices
# main entry point to create voices.json from characters.json
async def generate_voices(output_dir: Path) -> Path:
    characters_path = output_dir / "characters.json"
    voices_path = output_dir / "voices.json"
    if voices_path.exists():
        return voices_path
    if not characters_path.exists():
        raise ValueError("characters.json not found")
    characters = json.loads(characters_path.read_text(encoding="utf-8"))
    voices = await generate_all_voice_descriptions(characters)
    voices_path.write_text(json.dumps(voices, indent=2), encoding="utf-8")
    return voices_path


# ##################################################################
# generate voices sync
# synchronous wrapper for generate_voices
def generate_voices_sync(output_dir: Path) -> Path:
    return asyncio.run(generate_voices(output_dir))
