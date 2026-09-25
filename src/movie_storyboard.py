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
    roster = "\n".join(
        f"- {cid} ({info.get('name', cid)}): {(info.get('bio') or info.get('description') or '')[:600]}"
        for cid, info in characters.items()
    )
    prompt = f"""You are the character designer on an animated film. For each character below, write a tight VISUAL description (40-70 words) usable as an image-generation prompt: species/body, age appearance, face, hair, build, clothing, colors, distinguishing marks. No plot, no personality, no relationships — only what a viewer SEES. If the character has no visual form (e.g. a narrator), write "NONE".

Characters:
{roster}

Output JSON only: {{"<char_id>": "<visual description or NONE>", ...}}. No markdown, no explanation."""
    from src.voice_description import parse_json_response

    appearances = parse_json_response(ask_sync(prompt, max_tokens=4096))
    path.write_text(json.dumps(appearances, indent=2), encoding="utf-8")
    return appearances


# ##################################################################
# choose style
# one LLM call: a single consistent cinematic style anchor for every image
def choose_style(output_dir: Path) -> str:
    storyboard_path = output_dir / "storyboard.json"
    if storyboard_path.exists():
        return json.loads(storyboard_path.read_text(encoding="utf-8"))["style"]
    chapters = sorted((output_dir / "chapters").glob("*.txt"))
    sample = ""
    for p in chapters[1:]:
        sample = p.read_text(encoding="utf-8")[:2000]
        if len(sample.split()) > 100:
            break
    prompt = f"""You are the art director on an animated film adaptation of this story. Write ONE visual style paragraph (30-50 words) that will be appended to every image-generation prompt so the whole film looks consistent: medium (e.g. painterly digital illustration), palette, lighting, mood, level of detail. Never mention text, captions, or watermarks.

Story sample:
{sample}

Output the style paragraph only. No preamble."""
    return ask_sync(prompt, max_tokens=300)


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
