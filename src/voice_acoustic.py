import json
from pathlib import Path

from daz_agent_sdk import Conversation, Tier

SYSTEM_PROMPT = """You convert character biographies into terse acoustic voice descriptions for the qwen3-tts voice-design model.

Rules from the qwen3-tts documentation:
- 1 to 3 sentences, ideally 15-40 words total.
- Describe ACOUSTIC qualities only: gender, age, pitch, timbre, accent, pace, emotion, distinctive vocal traits.
- DO NOT include personality, biography, plot role, relationships, philosophy, what they say, or their background story.
- Be specific: "deep", "crisp", "fast-paced", "soft and breathy" — NOT "nice", "sophisticated".
- Combine multiple dimensions: gender + age + accent + pitch + timbre + pace.
- AVOID conflicting attributes (e.g. "high-pitched deep bass" — pick one).
- No celebrity names.
- English text only. The output speech may be in any language but the instruct text must be English.

You will be given one character's bio at a time. Reply with ONLY the acoustic description — no preamble, no quotes, no explanation. Just the description text."""


# ##################################################################
# generate acoustic for one
# convert one character bio into a tight acoustic description
async def generate_acoustic_descriptions(output_dir: Path) -> dict[str, str]:
    voices_path = output_dir / "voices.json"
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    async with Conversation(name="voice-acoustic", tier=Tier.HIGH, system=SYSTEM_PROMPT) as conv:
        for char_id, info in voices.items():
            bio = info.get("description") or info.get("bio") or ""
            prompt = f"Character id: {char_id}\nBiography:\n{bio}"
            resp = await conv.say(prompt, tier=Tier.HIGH, timeout=120)
            acoustic = resp.text.strip().strip('"').strip("'")
            out[char_id] = acoustic
            info["acoustic"] = acoustic
            print(f"  {char_id}: {acoustic}")
    voices_path.write_text(json.dumps(voices, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ##################################################################
# generate one
# helper for testing — process a single character and print result
async def generate_one(output_dir: Path, char_id: str) -> str:
    voices_path = output_dir / "voices.json"
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    if char_id not in voices:
        raise ValueError(f"{char_id} not in voices.json")
    bio = voices[char_id].get("description") or voices[char_id].get("bio") or ""
    async with Conversation(name="voice-acoustic-one", tier=Tier.HIGH, system=SYSTEM_PROMPT) as conv:
        resp = await conv.say(f"Character id: {char_id}\nBiography:\n{bio}", timeout=120)
    acoustic = resp.text.strip().strip('"').strip("'")
    print(f"\n=== {char_id} ===")
    print(f"BIO ({len(bio)} chars): {bio[:300]}{'...' if len(bio) > 300 else ''}")
    print(f"\nACOUSTIC ({len(acoustic)} chars): {acoustic}")
    return acoustic
