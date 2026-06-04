"""Kokoro voice catalog + LLM-driven character→voice mapping.

Kokoro has no voice cloning — it ships a fixed bank of named voice packs. We
therefore *map* each character's voice description onto the closest kokoro
voice (optionally a weighted blend of two) plus a speech-rate, using the
boringstack LLM. The result is saved to ``kokoro_voices.json``:

    {"<char_id>": {"voice": "af_heart" | "af_heart*0.6+am_michael*0.4",
                    "speed": 1.0, "description": "<original description>"}}

A "voice" string is exactly what the arbiter ``tts-kokoro`` job accepts.

Future (documented in arbiter/KOKORO_TTS.md): pre-evolve hundreds of custom
voices with kvoicewalk to grow the bank from 54 to ~500, then map onto that.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.llm import ask_sync

# Full kokoro voice bank (54). Language/gender/accent is fully determined by the
# id prefix: a=American EN, b=British EN, e=Spanish, f=French, h=Hindi,
# i=Italian, j=Japanese, p=Br-Portuguese, z=Mandarin; f=female, m=male.
ALL_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta",
    "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune",
    "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian",
    "zm_yunxi", "zm_yunxia", "zm_yunyang",
]

_LANG = {"a": "American English", "b": "British English", "e": "Spanish",
         "f": "French", "h": "Hindi", "i": "Italian", "j": "Japanese",
         "p": "Brazilian Portuguese", "z": "Mandarin Chinese"}

# Default English pool we ask the LLM to map onto (most books are English).
ENGLISH_VOICES = [v for v in ALL_VOICES if v[0] in ("a", "b")]
DEFAULT_VOICE = "af_heart"          # kokoro's reference / highest-quality voice
DEFAULT_NARRATOR = "bm_george"      # warm British male default for narration


# ##################################################################
# describe voice
# human-readable gender/accent for a voice id (certain from the prefix)
def describe_voice(voice_id: str) -> str:
    lang = _LANG.get(voice_id[0], "English")
    gender = "female" if voice_id[1] == "f" else "male"
    return f"{lang} {gender}"


# ##################################################################
# voice catalog text
# the menu we hand the LLM
def _catalog_text(voices: list[str]) -> str:
    return "\n".join(f"- {v}: {describe_voice(v)}" for v in voices)


# ##################################################################
# build prompt
# ask the LLM to assign every character a kokoro voice + speed in one call so
# voices stay distinct across the cast
def _build_prompt(characters: dict, voices: list[str]) -> str:
    cast = []
    for cid, info in characters.items():
        desc = info.get("description") or info.get("bio") or ""
        cast.append(f"- {cid}: {desc}")
    cast_text = "\n".join(cast)
    catalog = _catalog_text(voices)
    return f"""You assign each audiobook character a voice from the fixed Kokoro voice bank.

AVAILABLE KOKORO VOICES (id: accent + gender):
{catalog}

CHARACTERS (id: voice description):
{cast_text}

For EACH character choose:
- "voice": a voice id from the bank that best matches the character's gender and accent.
  You MAY blend two voices for a more distinctive timbre using the format
  "id1*0.6+id2*0.4" (weights are any positive numbers; same gender recommended).
- "speed": a speech-rate multiplier between 0.85 and 1.15 (older/graver = slower,
  younger/energetic = faster; 1.0 = normal).

Rules:
- Match gender exactly. Match accent when the description implies one (British,
  American, etc.); otherwise prefer American English voices.
- Make voices DISTINCT: do not give two different characters the same id+speed.
  Use different ids, blends, and speeds to separate same-gender characters.
- "narrator" should be a clear, pleasant, authoritative voice.
- Every character id MUST appear exactly once in the output.

Output ONLY valid JSON, no markdown, no commentary, exactly this shape:
{{"<char_id>": {{"voice": "<id or blend>", "speed": <number>}}}}"""


# ##################################################################
# parse mapping
# tolerant JSON extraction
def _parse_mapping(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        a = text.find("```")
        b = text.rfind("```")
        if a != b:
            block = text[a + 3:b].split("\n", 1)
            text = block[1] if len(block) > 1 else block[0]
    if not text.startswith("{"):
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1:
            text = text[i:j + 1]
    return json.loads(text)


# ##################################################################
# sanitize voice spec
# ensure every referenced voice id is real; fall back gracefully
def _sanitize(spec: str, fallback: str) -> str:
    spec = (spec or "").strip()
    if not spec:
        return fallback
    parts = []
    for part in spec.split("+"):
        name, _, w = part.partition("*")
        name = name.strip()
        if name not in ALL_VOICES:
            return fallback
        parts.append(f"{name}*{w.strip()}" if w.strip() else name)
    return "+".join(parts)


# ##################################################################
# map characters to voices
# produce kokoro_voices.json from voices.json using one LLM call
def map_characters_to_voices(output_dir: Path, voices: list[str] | None = None) -> Path:
    voices_path = output_dir / "voices.json"
    out_path = output_dir / "kokoro_voices.json"
    if out_path.exists():
        return out_path
    if not voices_path.exists():
        raise ValueError("voices.json not found — run voices_desc step first")
    characters = json.loads(voices_path.read_text(encoding="utf-8"))
    pool = voices or ENGLISH_VOICES

    raw = ask_sync(_build_prompt(characters, pool), temperature=0.4, max_tokens=4096)
    try:
        mapping = _parse_mapping(raw)
    except (json.JSONDecodeError, ValueError):
        mapping = {}

    result: dict[str, dict] = {}
    for cid, info in characters.items():
        entry = mapping.get(cid, {}) if isinstance(mapping, dict) else {}
        fallback = DEFAULT_NARRATOR if cid == "narrator" else DEFAULT_VOICE
        voice = _sanitize(entry.get("voice", ""), fallback)
        try:
            speed = float(entry.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.7, min(1.3, speed))
        result[cid] = {
            "voice": voice,
            "speed": round(speed, 3),
            "description": info.get("description") or info.get("bio") or "",
        }

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# ##################################################################
# load voice map
# read kokoro_voices.json → {char_id: (voice_spec, speed)}
def load_voice_map(output_dir: Path) -> dict[str, tuple[str, float]]:
    p = output_dir / "kokoro_voices.json"
    if not p.exists():
        raise ValueError("kokoro_voices.json not found — run voice mapping first")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {cid: (v["voice"], float(v.get("speed", 1.0))) for cid, v in data.items()}
