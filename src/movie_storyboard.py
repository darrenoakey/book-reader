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
# choose scene seconds
# per-book image frequency: <output>/scene_seconds.txt overrides
# TARGET_SECONDS (written by `./run create --scene-seconds N`); a rebuilt
# storyboard keeps its recorded value via storyboard.json scene spacing
def choose_scene_seconds(output_dir: Path) -> float:
    cfg = output_dir / "scene_seconds.txt"
    if cfg.exists():
        try:
            value = float(cfg.read_text(encoding="utf-8").strip())
        except ValueError:
            raise ValueError(f"scene_seconds.txt is not a number: {cfg.read_text().strip()!r}") from None
        if not 5.0 <= value <= 120.0:
            raise ValueError(f"scene_seconds {value} outside 5..120s")
        return value
    return TARGET_SECONDS


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

    def _mid_text(s: list[dict]) -> str:
        # The still hangs on screen for the WHOLE window, so it must depict
        # what is being said at the window's temporal MIDPOINT (a 5s scene is
        # judged at 2.5s), not the opening or closing beat.
        mid = (s[0]["start"] + s[-1]["end"]) / 2
        line = min(
            s,
            key=lambda l: (
                min(abs(mid - l["start"]), abs(mid - l["end"])) if not (l["start"] <= mid <= l["end"]) else 0.0
            ),
        )
        return line["text"]

    return [
        {
            "index": i,
            "start": round(s[0]["start"], 3),
            "end": round(s[-1]["end"], 3),
            "speakers": sorted({l["speaker"] for l in s}),
            "text": " ".join(l["text"] for l in s),
            "mid_text": _mid_text(s),
        }
        for i, s in enumerate(scenes)
    ]


# ##################################################################
# world bible
# one LLM call, cached: the world's GROUND RULES in plain visual language.
# The image generator has no idea what the book's terms mean — "dragonrider"
# drew HORSES until the prompts stated the rules. The bible also yields a
# forbidden-visuals list appended to every generation.
def world_bible(output_dir: Path, title: str) -> dict:
    path = output_dir / "world_bible.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    chapters = sorted((output_dir / "chapters").glob("*.txt"))
    sample = "\n\n".join(p.read_text(encoding="utf-8")[:3000] for p in chapters[1:3])
    prompt = f"""You are building the world bible for an animated film of "{title}" — the ground rules an IMAGE GENERATOR needs so it never contradicts the setting.

Read this sample of the book:
{sample[:5500]}

Output JSON only:
{{
  "world_summary": "<2-3 sentences: what kind of world, era, technology level, look>",
  "rules": ["<plain visual directives about how the world works — e.g. 'riders ride large winged dragons; people travel on dragonback or on foot'>, ..."],
  "forbidden": ["<things that must NEVER appear because they don't exist in this world or contradict it — e.g. 'horses', 'cars', 'guns'> — be aggressive: anything the image model might wrongly default to>"]
}}

Rules for the rules: state what things MEAN visually (an image model has not read the book); translate in-world terms into plain visuals; if the book's people ride dragons, say explicitly that there are no horses. No markdown."""
    from src.voice_description import parse_json_response

    bible = parse_json_response(ask_sync(prompt, max_tokens=1200))
    bible.setdefault("world_summary", "")
    bible.setdefault("rules", [])
    bible.setdefault("forbidden", [])
    path.write_text(json.dumps(bible, indent=2), encoding="utf-8")
    return bible


# ##################################################################
# distill locations
# one LLM call, cached: named LOCATIONS get the same treatment as characters —
# gather everything the text says about each recurring place and distill one
# canonical visual per location, so the Hatching Ground (etc.) looks the same
# every time it appears.
def distill_locations(output_dir: Path, bible: dict) -> dict:
    path = output_dir / "locations.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    chapters = sorted((output_dir / "chapters").glob("*.txt"))
    full = "\n\n".join(p.read_text(encoding="utf-8") for p in chapters[1:])
    prompt = f"""You are the production designer on an animated film. Locations are characters too: find every NAMED or clearly-recurring LOCATION in this text (buildings, rooms, caverns, grounds, halls — places the action returns to), and for each, harvest EVERY visual detail the text gives and distill ONE canonical visual description.

World rules (respect them): {bible.get("world_summary", "")} {" ".join(bible.get("rules", []))}

Text:
{full[:12000]}

For each location output:
- id: lowercase_snake (e.g. "hatching_ground", "great_hall", "lower_corridors")
- name: display name
- description: 50-80 words, CONCRETE and self-contained: size, materials, colors, light, key features, mood — what a viewer SEES. No in-world jargon without a plain explanation, no plot, no characters. Merge every detail the text gives about the place into one consistent picture.

Output JSON only: {{"<loc_id>": {{"name": "...", "description": "..."}}, ...}}. No markdown."""
    from src.voice_description import parse_json_response

    locations = parse_json_response(ask_sync(prompt, max_tokens=3000))
    path.write_text(json.dumps(locations, indent=2), encoding="utf-8")
    return locations


def distill_appearances(output_dir: Path, bible: dict | None = None) -> dict:
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

{_bible_block(bible)}HARD RULES:
- The image generator has NEVER read the book. The description must be fully self-contained: no names from the story world, no in-world terms, no plot, no roles, no relationships.
- Source details are AUTHORITATIVE: anything the source text states (footwear, clothing, hair, marks) MUST appear in your description unchanged — never drop a stated detail, never replace it with a stereotype. (A book said "heavy wher-hide boots"; a previous draft wrote "barefoot". Never again.)
- Be SPECIFIC and CONCRETE: age in years, height/build, hair color and style, eye color, face, skin tone, clothing described by plain garment names and colors.
- Humans are plain humans: normal skin, no scales, no fur, no animal features, unless the source text explicitly describes them.
- Describe the character's CANONICAL default appearance only. Temporary states are NOT part of the look: ignore injuries, bandages, casts, dirt, disguises, bedding, or items carried in one scene, unless the character has them for essentially the whole story.
- Defaults are ONLY for gaps the source never mentions, and must be SIMPLE and neutral, consistent with what IS given (age, gender, build) — never exotic.
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
# bible block
# render the world bible as a prompt preamble (used by every LLM visual step)
def _bible_block(bible: dict) -> str:
    if not bible:
        return ""
    summary = bible.get("world_summary") or bible.get("context") or ""
    if not summary:
        return ""
    rules = "\n".join(f"- {r}" for r in bible.get("rules", []))
    forbidden = ", ".join(bible.get("forbidden", []))
    return (
        f"WORLD CONTEXT (ground truth about this story's world — overrides your assumptions):\n"
        f"{summary}\n{rules}\n"
        f"NEVER show (does not exist in this world): {forbidden}\n\n"
    )


# ##################################################################
# prompt for scene
# one LLM call per scene: the window's spoken text → a cinematic still prompt
def _scene_prompt(scene: dict, style: str, appearances: dict, title: str, bible: dict | None = None) -> dict:
    cast_notes = []
    for speaker in scene["speakers"]:
        if speaker == "narrator":
            continue
        appearance = appearances.get(speaker, "")
        if appearance and appearance != "NONE":
            cast_notes.append(f"{speaker}: {appearance}")
    cast_block = "\n".join(cast_notes) or "(no named characters speak in this window)"
    prompt = f"""You are the storyboard artist on an animated film of "{title}". Write an image prompt for ONE cinematic 16:9 still illustrating this {scene["end"] - scene["start"]:.0f}-second moment of the story.

{_bible_block(bible)}The still stays on screen for the whole window, so it MUST depict the window's temporal midpoint. Illustrate THIS moment:
{scene.get("mid_text") or scene["text"][:400]}

Full spoken text during the window (context only — the midpoint moment above is the subject):
{scene["text"][:1200]}

Characters speaking in this window (show only characters who are actually present in the action; use these exact visual descriptions if you show them):
{cast_block}

Rules: describe the SCENE (setting, action, composition, lighting, camera framing). Do not mention sound, narration, or dialogue. No text in the image. Show at most the listed characters. If no listed character is present, depict the setting/action alone. NEVER use negation ("no X", "never X", "without X") — describe only what IS present; exclusions are enforced through a separate channel, and negated words leak the concept into the image.

TWO HARD RULES about consistency:
- Every character you NAME anywhere in the prompt MUST also appear in the "characters" list — the list drives which reference portraits condition the image, so a named-but-unlisted character renders as a random stranger.
- Wardrobe lock: restate each shown character's clothing from their description and NEVER dress characters in matching/coordinated outfits unless their descriptions say so. A uniform described for ONE character belongs to that character alone.
- Each character appears EXACTLY ONCE in the scene — the prompt must never place the same person in two spots, and must not describe a crowd that could include them.

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
    windows = window_lines(lines, target_seconds=choose_scene_seconds(output_dir))
    style = choose_style(output_dir)
    bible = world_bible(output_dir, title)
    appearances = distill_appearances(output_dir, bible)
    locations = distill_locations(output_dir, bible)
    print(f"  storyboard: {len(windows)} scenes, style: {style[:80]}...")
    scenes = []
    for window in windows:
        scenes.append(_scene_prompt(window, style, appearances, title, bible))
        if (window["index"] + 1) % 10 == 0:
            print(f"    {window['index'] + 1}/{len(windows)} scene prompts")
    storyboard_path.write_text(
        json.dumps(
            {
                "style": style,
                "appearances": appearances,
                "locations": locations,
                "world_bible": bible,
                "scenes": scenes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return storyboard_path
