"""Movie images: character reference portraits and per-scene stills.

Uses the owner-sanctioned arbiter ``qwen-image`` job type (Qwen-Image-2.1,
local on spark). Two artifact kinds:

  refs/<char_id>.png    one identity portrait per visually-described
                        character, generated from the appearance description
                        distilled by the storyboard step. These are the
                        canonical likenesses every scene image conditions on.
  scenes/NNNN.png       one 16:9 cinematic still per storyboard scene,
                        generated in edit mode with the speaking characters'
                        portraits as reference input so faces/outfits stay
                        consistent across the whole film.

The qwen-image adapter accepts ONE condition image per job, so scenes with
several characters get a single labelled contact sheet (portraits side by
side, name captions burned in) — the model reads it as a character lineup
and the prompt tells it exactly that.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PIL import Image, ImageDraw

from src.arbiter_tts import ARBITER_BASE, ArbiterClient, ArbiterError

log = logging.getLogger(__name__)

REF_SIZE = 1024
SCENE_WIDTH = 1920
SCENE_HEIGHT = 1088  # 16:9 snapped to /16 as the adapter requires
RESTRAINT = (
    " Clean, well-composed cinematic still. Anatomically correct people. "
    "No extra limbs. No text, no captions, no watermark."
)


# ##################################################################
# submit image
# one qwen-image job → PNG bytes at output_path; retries forever like the
# TTS helpers (the pipeline must never silently lose work)
def qwen_image_to_file(
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    ref_image: Path | None = None,
    seed: int = 42,
    steps: int = 40,
    why: str | None = None,
) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 10000:
        return output_path
    from arbiter_client import stage_file

    client = ArbiterClient(base_url=ARBITER_BASE, timeout=120)
    params: dict = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "force": True,
    }
    if ref_image is not None:
        params["image_file"] = stage_file(ref_image)
    jid = client.submit("qwen-image", who="book-reader", why=why or output_path.name, **params)
    while True:
        try:
            client.poll(jid, interval=3.0, timeout=31536000)
            data = client.get_result_bytes(jid)
            if len(data) >= 10000:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(data)
                return output_path
            log.warning("qwen-image %s empty result — resubmitting", output_path.name)
            jid = client.submit("qwen-image", who="book-reader", why=why or output_path.name, **params)
        except ArbiterError as e:
            msg = str(e).lower()
            if "failed" in msg or "cancelled" in msg or "timed out" in msg:
                log.warning("qwen-image %s job died (%s) — resubmitting", output_path.name, e)
                if ref_image is not None:
                    params["image_file"] = stage_file(ref_image)
                jid = client.submit("qwen-image", who="book-reader", why=why or output_path.name, **params)
            else:
                log.warning("qwen-image %s transient (%s) — retrying poll", output_path.name, e)
                time.sleep(5)
        except (ConnectionError, OSError) as e:
            log.warning("qwen-image %s connection (%s) — retrying poll", output_path.name, e)
            time.sleep(5)


# ##################################################################
# contact sheet
# compose several character portraits into ONE labelled lineup image — the
# qwen-image adapter takes a single condition image, so multi-character
# scenes reference a strip of portraits with name captions
def build_contact_sheet(ref_paths: list[tuple[str, Path]], output_path: Path, tile: int = 512) -> Path:
    label_h = 56
    sheet = Image.new("RGB", (tile * len(ref_paths), tile + label_h), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (char_id, path) in enumerate(ref_paths):
        portrait = Image.open(path).convert("RGB").resize((tile, tile))
        sheet.paste(portrait, (i * tile, 0))
        name = char_id.replace("-", " ").title()
        draw.rectangle([i * tile, tile, (i + 1) * tile, tile + label_h], fill=(24, 24, 28))
        draw.text((i * tile + 12, tile + 18), name, fill=(235, 235, 240))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path


# ##################################################################
# generate character refs
# one identity portrait per visually-described character (narrator/NONE skip)
def generate_character_refs(output_dir: Path) -> list[Path]:
    storyboard = json.loads((output_dir / "storyboard.json").read_text(encoding="utf-8"))
    appearances = storyboard["appearances"]
    style = storyboard["style"]
    refs_dir = output_dir / "refs"
    written: list[Path] = []
    characters = json.loads((output_dir / "characters.json").read_text(encoding="utf-8"))
    for index, (char_id, appearance) in enumerate(sorted(appearances.items())):
        if not appearance or appearance == "NONE":
            continue
        dest = refs_dir / f"{char_id}.png"
        name = characters.get(char_id, {}).get("name", char_id)
        prompt = (
            f"Head-and-shoulders identity portrait of {name}. {appearance} "
            f"Looking at camera, face large and unmistakable, neutral expression, plain dark background. "
            f"Style: {style}.{RESTRAINT}"
        )
        qwen_image_to_file(prompt, dest, REF_SIZE, REF_SIZE, seed=7000 + index, why=f"character ref {char_id}")
        written.append(dest)
    return written


# ##################################################################
# scene ref sheet
# choose the condition image for one scene: the single character's portrait,
# or a contact sheet of all shown characters; None when the scene shows no
# named character
def _scene_ref_image(output_dir: Path, scene: dict, available: dict[str, Path]) -> Path | None:
    chars = [c for c in scene.get("characters", []) if c in available]
    if not chars:
        return None
    if len(chars) == 1:
        return available[chars[0]]
    sheet = output_dir / "refs" / f"sheet_{'_'.join(sorted(chars))}.png"
    if not sheet.exists():
        build_contact_sheet([(c, available[c]) for c in sorted(chars)], sheet)
    return sheet


# ##################################################################
# generate scene images
# one cinematic still per storyboard scene, conditioned on the speaking
# characters' reference portraits for visual consistency
def generate_scene_images(output_dir: Path) -> list[Path]:
    storyboard = json.loads((output_dir / "storyboard.json").read_text(encoding="utf-8"))
    scenes = storyboard["scenes"]
    refs_dir = output_dir / "refs"
    available = {p.stem: p for p in refs_dir.glob("*.png") if not p.name.startswith("sheet_")}
    scenes_dir = output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scene in scenes:
        dest = scenes_dir / f"{scene['index']:04d}.png"
        ref = _scene_ref_image(output_dir, scene, available)
        prompt = scene["prompt"]
        if ref is not None:
            if ref.name.startswith("sheet_"):
                prompt = (
                    "The reference image is a labelled lineup of character identity portraits. "
                    "Draw these exact characters, keeping their faces, hair, clothing and colors faithful. "
                ) + prompt
            else:
                prompt = (
                    "The reference image is an identity portrait of the main character in this scene. "
                    "Keep their face, hair, clothing and colors faithful. "
                ) + prompt
        qwen_image_to_file(
            prompt,
            dest,
            SCENE_WIDTH,
            SCENE_HEIGHT,
            ref_image=ref,
            seed=42000 + scene["index"],
            why=f"scene {scene['index']:04d}",
        )
        written.append(dest)
        if (scene["index"] + 1) % 5 == 0:
            print(f"    {scene['index'] + 1}/{len(scenes)} scene images")
    return written
