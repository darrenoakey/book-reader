import asyncio
import json
from pathlib import Path

from daz_agent_sdk import Tier, agent


# ##################################################################
# query sonnet
# send a prompt to claude sonnet and get text response
async def query_sonnet(prompt: str) -> str:
    response = await agent.ask(prompt, tier=Tier.MEDIUM)
    return response.text.strip()


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


# ##################################################################
# generate all voice descriptions
# create voice descriptions for all characters in one call
async def generate_all_voice_descriptions(characters: dict) -> dict:
    chars_json = json.dumps(characters, indent=2)
    prompt = f"""Create voice descriptions for audiobook text-to-speech voice cloning.

Input characters JSON:
{chars_json}

For each character, create a detailed voice description suitable for AI TTS voice cloning.
Include:
- Gender and approximate age
- Vocal qualities (pitch, timbre, resonance, rasp, etc.)
- Speaking style (pace, emphasis, warmth, formality)
- Accent or dialect that fits their background
- Emotional tone and personality in voice

For the "narrator" character, focus on audiobook narration qualities.
Keep each description 60-100 words. Be specific and vivid.

Return ONLY valid JSON in this exact format:
{{
  "character_id": {{
    "description": "The voice description text"
  }}
}}

Include ALL characters from the input. Use the same character IDs as keys."""

    response = await query_sonnet(prompt)
    return parse_json_response(response)


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
