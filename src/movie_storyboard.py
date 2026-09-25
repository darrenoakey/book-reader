"""Movie storyboard: segment the audiobook into ~30s scenes and write a
cinematic image prompt for each.

Inputs (produced by earlier pipeline steps):
  audio/*.timeline.json   per-line speaker + start/end times per chapter
  audio/*.wav             chapter audio (for chapter offsets)
  characters.json         character bios (for appearance distillation)

Output:
  storyboard.json         {style, appearances: {id: visual description},
                           scenes: [{index, start, end, text_excerpt,
                                     characters: [ids], prompt}]}

Scene boundaries snap to spoken-line boundaries: accumulate whole lines until
the window reaches TARGET_SECONDS (overshooting to finish the current line is
fine — the picture should never change mid-sentence).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.audio_synth import wav_duration
from src.llm import ask_sync

TARGET_SECONDS = 30.0


# ##################################################################
# load global lines
# flatten every chapter timeline into one global line stream with absolute
# second offsets across the whole book
def load_global_lines(output_dir: Path) -> list[dict]:
    audio_dir = output_dir / "audio"
    timelines = sorted(audio_dir.glob("*.timeline.json"))
    if not timelines:
        raise ValueError("no audio timelines — run the audio step first")
    lines: list[dict] = []
    offset = 0.0
    for tl_path in timelines:
        chapter_wav = audio_dir / f"{tl_path.stem.replace('.timeline', '')}.wav"
        chapter = json.loads(tl_path.read_text(encoding="utf-8"))
        for entry in chapter["lines"]:
            lines.append(
                {
                    "speaker": entry["speaker"],
                    "text": entry["text"],
                    "start": entry["start"] + offset,
                    "end": entry["end"] + offset,
                    "chapter": chapter["chapter"],
                }
            )
        offset += wav_duration(chapter_wav)
    return lines


# ##################################################################
# window lines
# group the global line stream into scenes of about target_seconds
def window_lines(lines: list[dict], target_seconds: float = TARGET_SECONDS) -> list[dict]:
    scenes: list[dict] = []
    current: list[dict] = []
    for line in lines:
        current.append(line)
        if current[-1]["end"] - current[0]["start"] >= target_seconds:
            scenes.append(current)
            current = []
    if current:
        if scenes and current[-1]["end"] - current[0]["start"] < target_seconds / 2:
            scenes[-1].extend(current)  # tail too short — fold into previous
        else:
            scenes.append(current)
    return [
        {
            "index": i,
            "start": round(s[0]["start"], 3),
            "end": round(s[-1]["end"], 3),
            "speakers": sorted({l["speaker"] for l in s}),
            "text": " ".join(l["text"] for l in s),
        }
        for i, s in enumerate(scenes)
    ]


# ##################################################################
# distill appearances
# one LLM call: characters.json bios → tight visual descriptions for image
# generation (image models need LOOKS, not plot roles)
def distill_appearances(output_dir: Path) -> dict:
    path = output_dir / "appearances.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    characters = json.loads((output_dir / "characters.json").read_text(encoding="utf-8"))
    # Prefer the dedicated `look` field (everything the text says about
    # appearance); fall back to the voice bio only when no look was captured.
    roster = "\n".join(
        f"- {cid} ({info.get('name', cid)}): {((info.get('look') or '').strip() or (info.get('bio') or info.get('description') or ''))[:800]}"
        for cid, info in characters.items()
    )
    prompt = f"""You are the character designer on an animated family film. For each character below, write a tight VISUAL description (50-80 words) for an image generator.

HARD RULES:
- The image generator has NEVER read the book. The description must be fully self-contained: no names from the story world, no in-world terms, no plot, no roles, no relationships.
- Be SPECIFIC and CONCRETE: age in years, height/build, hair color and style, eye color, face, skin tone, clothing described by plain garment names and colors. Less "young fantasy boy", more "boy, 10 years old, short and slight for his age, messy brown hair, brown eyes, loose beige linen shirt and brown trousers, bare feet".
- Humans are plain humans: normal skin, no scales, no fur, no animal features, unless the source text explicitly describes them.
- If the source gives few visual details, fill in SIMPLE neutral defaults consistent with what is given (age, gender, build) — never exotic ones.
- If the character has no visual form (e.g. a narrator), write "NONE".

Characters:
{roster}

Output JSON only: {{"<char_id>": "<visual description or NONE>", ...}}. No markdown, no explanation."""
    from src.voice_description import parse_json_response

    appearances = parse_json_response(ask_sync(prompt, max_tokens=4096))
    path.write_text(json.dumps(appearances, indent=2), encoding="utf-8")
    return appearances


# ##################################################################
# choose style
# the film's single visual style. DEFAULT is bright/happy/cartoonish (owner
# rule 2026-09-26): an LLM-invented style once picked "dramatic chiaroscuro,
# textures of stone and scales" for a hopeful children's tale — grimdark
# scenes and literal scales on human skin. The style is now a deterministic
# SETTING, never an LLM invention. Override order: $BOOK_MOVIE_STYLE, then
# <output>/style.txt, then DEFAULT_STYLE.
DEFAULT_STYLE = (
    "Bright, cheerful animated-family-film cartoon: bold clean shapes, warm saturated "
    "colors, sunny optimistic lighting, soft cel shading, friendly expressive faces, "
    "storybook charm. Light and hopeful even in tense moments."
)


def choose_style(output_dir: Path) -> str:
    storyboard_path = output_dir / "storyboard.json"
    if storyboard_path.exists():
        return json.loads(storyboard_path.read_text(encoding="utf-8"))["style"]
    env = os.environ.get("BOOK_MOVIE_STYLE")
    if env and env.strip():
        return env.strip()
    style_txt = output_dir / "style.txt"
    if style_txt.exists():
        return style_txt.read_text(encoding="utf-8").strip()
    return DEFAULT_STYLE


# ##################################################################
# prompt for scene
# one LLM call per scene: the window's spoken text → a cinematic still prompt
def _scene_prompt(scene: dict, style: str, appearances: dict, title: str) -> dict:
    cast_notes = []
    for speaker in scene["speakers"]:
        if speaker == "narrator":
            continue
        appearance = appearances.get(speaker, "")
        if appearance and appearance != "NONE":
            cast_notes.append(f"{speaker}: {appearance}")
    cast_block = "\n".join(cast_notes) or "(no named characters speak in this window)"
    prompt = f"""You are the storyboard artist on an animated film of "{title}". Write an image prompt for ONE cinematic 16:9 still illustrating this {scene['end'] - scene['start']:.0f}-second moment of the story.

Spoken text during the window:
{scene['text'][:1200]}

Characters speaking in this window (show only characters who are actually present in the action; use these exact visual descriptions if you show them):
{cast_block}

Rules: describe the SCENE (setting, action, composition, lighting, camera framing). Do not mention sound, narration, or dialogue. No text in the image. Show at most the listed characters. If no listed character is present, depict the setting/action alone.

TWO HARD RULES about consistency:
- Every character you NAME anywhere in the prompt MUST also appear in the "characters" list — the list drives which reference portraits condition the image, so a named-but-unlisted character renders as a random stranger.
- Wardrobe lock: restate each shown character's clothing from their description and NEVER dress characters in matching/coordinated outfits unless their descriptions say so. A uniform described for ONE character belongs to that character alone.

Output JSON only: {{"prompt": "<60-100 word image prompt>", "characters": ["<char_ids actually shown>"]}}. No markdown."""
    from src.voice_description import parse_json_response

    result = parse_json_response(ask_sync(prompt, max_tokens=600))
    return {
        "index": scene["index"],
        "start": scene["start"],
        "end": scene["end"],
        "characters": [c for c in result.get("characters", []) if c in appearances],
        "prompt": f"{result['prompt'].strip()} Style: {style}",
        "text_excerpt": scene["text"][:200],
    }


# ##################################################################
# build storyboard
# full step: timelines → storyboard.json (skips if already built)
def build_storyboard(output_dir: Path, title: str) -> Path:
    storyboard_path = output_dir / "storyboard.json"
    if storyboard_path.exists():
        return storyboard_path
    lines = load_global_lines(output_dir)
    windows = window_lines(lines)
    style = choose_style(output_dir)
    appearances = distill_appearances(output_dir)
    print(f"  storyboard: {len(windows)} scenes, style: {style[:80]}...")
    scenes = []
    for window in windows:
        scenes.append(_scene_prompt(window, style, appearances, title))
        if (window["index"] + 1) % 10 == 0:
            print(f"    {window['index'] + 1}/{len(windows)} scene prompts")
    storyboard_path.write_text(
        json.dumps({"style": style, "appearances": appearances, "scenes": scenes}, indent=2),
        encoding="utf-8",
    )
    return storyboard_path
