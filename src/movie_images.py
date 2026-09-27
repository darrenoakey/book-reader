"""Movie stills: character reference portraits and per-scene cinematic images
via the arbiter ``qwen-image`` job type (Qwen-Image-2.1 on spark).

Consistency strategy: generate ONE identity portrait per named character
first (``refs/<char_id>.png``), then every scene image is an EDIT job
conditioned on that character's portrait — or on a labelled contact sheet of
portraits when several characters share the scene — so faces and costumes
stay stable across the whole film.

Result shape gotcha: the qwen-image adapter returns the FILE-ONLY result
shape (``{"file": "result.png"}``, no inline data, no result_path), so bytes
are fetched through the CIFS mount mapping
``/mnt/arbiter-store/...`` → ``/Volumes/ssd_4/arbiter/...`` with a
``get_result_bytes`` fallback for the inline shape.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PIL import Image, ImageDraw

from src.arbiter_tts import _client, _submit

log = logging.getLogger(__name__)

# Scene stills are 16:9, both dimensions snapped to /16 (model requirement).
SCENE_WIDTH = 1920
SCENE_HEIGHT = 1088
REF_SIZE = 1024
LABEL_BAND = 56  # px of label strip under each contact-sheet tile

RESTRAINT = (
    " No text, no captions, no watermark, no extra limbs, anatomically correct."
    " Human characters have completely normal human skin — no scales, no reptilian"
    " patches, no dragon features — unless the character description explicitly says so."
    " Output ONE single continuous cinematic scene — never panels, split screen,"
    " collage, borders, or labels."
    " Each named character appears EXACTLY ONCE in the scene — never show the same"
    " person twice; background people are clearly different individuals."
)

# Spark's /mnt/arbiter-store is this Mac's /Volumes/ssd_4/arbiter (CIFS).
_SPARK_PREFIX = "/mnt/arbiter-store/"
_LOCAL_PREFIX = "/Volumes/ssd_4/arbiter/"


# ##################################################################
# mount resolve
# map a spark-side /mnt/arbiter-store path to the local CIFS mount
def _mount_resolve(spark_path: str) -> Path | None:
    if spark_path.startswith(_SPARK_PREFIX):
        local = Path(_LOCAL_PREFIX + spark_path[len(_SPARK_PREFIX):])
        if local.exists():
            return local
    return None


# ##################################################################
# fetch image
# poll a qwen-image job and write its PNG to dest; resubmit only when the
# job is genuinely dead (never on a slow poll — that just re-queues it)
def _fetch_image(client, job_id: str, params: dict, dest: Path, why: str) -> None:
    from arbiter_client import ArbiterError

    current = job_id
    while True:
        try:
            status = client.poll(current, interval=2.0, timeout=31536000)
            result = status.get("result", {}) if isinstance(status, dict) else {}
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = b""
            if result.get("data") or result.get("result_path"):
                try:
                    data = client.get_result_bytes(current)
                except ArbiterError:
                    data = b""
            if not data and result.get("file"):
                local = _mount_resolve(f"/mnt/arbiter-store/output/jobs/{current}/{result['file']}")
                if local:
                    data = local.read_bytes()
            if len(data) >= 1000:
                dest.write_bytes(data)
                return
            log.warning("qwen-image %s for %s: empty result — resubmitting", current, dest.name)
            current = _submit(client, "qwen-image", params, why=why)
        except ArbiterError as e:
            msg = str(e).lower()
            if "failed" in msg or "cancelled" in msg or "timed out" in msg or "not found" in msg:
                log.warning("qwen-image %s died (%s) — resubmitting", current, e)
                current = _submit(client, "qwen-image", params, why=why)
            else:
                log.warning("qwen-image %s transient (%s) — retrying poll", current, e)
                time.sleep(5)
        except (ConnectionError, OSError) as e:
            log.warning("qwen-image %s connection (%s) — retrying poll", current, e)
            time.sleep(5)


# ##################################################################
# qwen image to file
# one qwen-image job (t2i, or edit when ref_image given) → dest PNG
def qwen_image_to_file(
    prompt: str,
    dest: Path,
    width: int,
    height: int,
    seed: int = 42,
    steps: int = 40,
    ref_image: Path | None = None,
    why: str | None = None,
    negative_prompt: str | None = None,
) -> Path:
    if dest.exists() and dest.stat().st_size >= 1000:
        return dest
    from arbiter_client import stage_file

    client = _client(120)
    clean: dict = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "force": True,
    }
    if negative_prompt:
        # The adapter only applies negative_prompt with true_cfg_scale > 1.
        # World-forbidden items belong HERE, never in the positive prompt —
        # writing "NO horses" into a prompt makes the model paint horses.
        clean["negative_prompt"] = negative_prompt
        clean["true_cfg_scale"] = 2.5
    if ref_image is not None:
        clean["image_file"] = stage_file(ref_image)
    reason = why or f"image {dest.name}"
    jid = _submit(client, "qwen-image", clean, why=reason)
    _fetch_image(client, jid, clean, dest, reason)
    return dest


# ##################################################################
# build reference sheet
# compose the condition image for a scene job as a CHARACTER REFERENCE SHEET:
# dark canvas, a title band that declares the image's purpose, and labelled
# portrait tiles with generous margins. A bare side-by-side strip reads as a
# comic page and the model reproduces the panels in the scene (observed: a
# 3-tile strip became a 3-panel triptych); the reference-sheet framing tells
# it "this is lookup material, not a layout".
TITLE_BAND = 72
_MARGIN = 28
_GAP = 20


def build_contact_sheet(items: list[tuple[str, Path]], dest: Path, tile: int = REF_SIZE // 2) -> Path:
    if not items:
        raise ValueError("reference sheet needs at least one image")
    dest.parent.mkdir(parents=True, exist_ok=True)
    width = _MARGIN * 2 + tile * len(items) + _GAP * (len(items) - 1)
    height = TITLE_BAND + tile + LABEL_BAND + _MARGIN * 2
    sheet = Image.new("RGB", (width, height), (24, 24, 32))
    draw = ImageDraw.Draw(sheet)
    title = "CHARACTER REFERENCE SHEET - portraits for likeness only - NOT a scene, NOT a layout"
    tb = draw.textbbox((0, 0), title)
    draw.text(((width - (tb[2] - tb[0])) // 2, (TITLE_BAND - (tb[3] - tb[1])) // 2), title, fill=(200, 200, 210))
    for i, (name, path) in enumerate(items):
        img = Image.open(path).convert("RGB")
        if img.width != img.height:  # centre-crop to square
            side = min(img.width, img.height)
            img = img.crop(((img.width - side) // 2, (img.height - side) // 2, (img.width + side) // 2, (img.height + side) // 2))
        x0 = _MARGIN + i * (tile + _GAP)
        sheet.paste(img.resize((tile, tile), Image.LANCZOS), (x0, TITLE_BAND))
        label = name.replace("-", " ").title()
        bbox = draw.textbbox((0, 0), label)
        lx = x0 + (tile - (bbox[2] - bbox[0])) // 2
        draw.text((lx, TITLE_BAND + tile + (LABEL_BAND - (bbox[3] - bbox[1])) // 2), label, fill=(235, 235, 235))
    sheet.save(dest, format="PNG")
    return dest


# ##################################################################
# load storyboard
# read storyboard.json, raising a clear error when missing
def _load_storyboard(output_dir: Path) -> dict:
    path = output_dir / "storyboard.json"
    if not path.exists():
        raise ValueError("storyboard.json not found — run the storyboard step first")
    return json.loads(path.read_text(encoding="utf-8"))


# ##################################################################
# generate character refs
# one identity portrait per visually-described character → refs/<id>.png
def generate_character_refs(output_dir: Path) -> list[Path]:
    storyboard = _load_storyboard(output_dir)
    style = storyboard["style"]
    appearances = storyboard["appearances"]
    characters_path = output_dir / "characters.json"
    names = {}
    if characters_path.exists():
        characters = json.loads(characters_path.read_text(encoding="utf-8"))
        names = {cid: (info or {}).get("name", cid) for cid, info in characters.items()}
    refs_dir = output_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cast = [(cid, a) for cid, a in sorted(appearances.items()) if a and a != "NONE" and cid != "narrator"]
    for index, (cid, appearance) in enumerate(cast):
        dest = refs_dir / f"{cid}.png"
        name = names.get(cid, cid.replace("-", " "))
        prompt = (
            f"Head-and-shoulders character reference portrait of {name}. {appearance}. "
            f"Facing camera, full face visible, neutral soft-lit background, face large and unmistakable. "
            f"Style: {style}.{RESTRAINT}"
        )
        print(f"  ref {index + 1}/{len(cast)}: {cid}")
        qwen_image_to_file(prompt, dest, REF_SIZE, REF_SIZE, seed=5000 + index, why=f"character ref {cid}")
        written.append(dest)
    return written


# ##################################################################
# scene ref
# pick the conditioning image for one scene: the single character's portrait,
# or a labelled contact sheet when several characters appear together
# ##################################################################
# scene condition
# the single condition image for one scene job (qwen-image takes ONE image):
# a labelled strip whose FIRST tile is the previous scene (style continuity —
# without it, scenes drift between photo, painting, and cartoon looks) and
# whose remaining tiles are the visible characters' portraits (identity).
# Scene 0 (no previous) uses portraits alone; a character-less scene N>0
# conditions on the previous scene alone.
def _scene_condition(output_dir: Path, characters: list[str], index: int) -> tuple[Path | None, str]:
    refs_dir = output_dir / "refs"
    available = [(cid, refs_dir / f"{cid}.png") for cid in characters if (refs_dir / f"{cid}.png").exists()]
    prev = output_dir / "scenes" / f"{index - 1:04d}.png"
    has_prev = index > 0 and prev.exists() and prev.stat().st_size >= 1000

    style_note = (
        "The reference sheet's FIRST tile (labelled STYLE REF) is the previous scene: "
        "match its art style, palette, and lighting EXACTLY — same medium, same rendering. "
        "Paint a NEW scene, do not copy its composition. "
    ) if has_prev else ""

    if not available:
        if not has_prev:
            return None, ""
        return prev, (
            "The reference image is the previous scene: match its art style, palette, and "
            "lighting EXACTLY — same medium, same rendering. Paint a NEW scene, do not copy "
            "its composition. "
        )

    if not has_prev and len(available) == 1:
        cid, path = available[0]
        return path, f"The reference image is a portrait of {cid.replace('-', ' ')}; keep this exact face, hair, and clothing."

    tiles: list[tuple[str, Path]] = ([("STYLE REF", prev)] if has_prev else []) + available
    sheet = build_contact_sheet(tiles, output_dir / "scenes" / ".sheets" / f"{index:04d}.png")
    names = ", ".join(cid.replace("-", " ") for cid, _ in available)
    return sheet, (
        f"{style_note}The reference image is a CHARACTER REFERENCE SHEET: separate labelled "
        f"portrait panels ({names}) for likeness lookup only. Do NOT reproduce its layout — "
        "paint ONE continuous cinematic scene using these exact faces, hair, and clothing. "
    )


# ##################################################################
# panel borders
# detect a panelized output (the model reproducing the reference sheet's
# layout as side-by-side panels). A panel border is a strong vertical edge
# sustained in >65% of rows with a full-height brightness step — ordinary
# scene content (columns, doorways) doesn't hold an edge that uniformly.
def panel_borders(path: Path) -> list[int]:
    import numpy as np

    img = np.asarray(Image.open(path).convert("L").resize((640, 360)), dtype=np.float32)
    dx = np.abs(np.diff(img, axis=1))
    strong = (dx > 35).mean(axis=0)
    step = []
    for x in range(8, img.shape[1] - 9):
        step.append(abs(img[:, x - 8 : x].mean() - img[:, x + 1 : x + 9].mean()))
    step = [0.0] * 8 + step + [0.0] * 8
    hits = [x for x in range(img.shape[1] - 1) if strong[x] > 0.65 and step[x] > 12]
    merged: list[list[int]] = []
    for x in hits:
        if not merged or x - merged[-1][-1] > 5:
            merged.append([x])
        else:
            merged[-1].append(x)
    return [int(sum(g) / len(g)) for g in merged]


# ##################################################################
# forbidden negative
# the world-bible forbidden list for the NEGATIVE prompt channel — putting
# "no horses" in the positive prompt is how we GOT horses (negation tokens
# attract the concept)
def _forbidden_negative(storyboard: dict) -> str:
    forbidden = (storyboard.get("world_bible") or {}).get("forbidden", [])
    return ", ".join(forbidden)


# ##################################################################
# generate scene images
# one cinematic 16:9 still per storyboard scene → scenes/NNNN.png
def generate_scene_images(output_dir: Path) -> list[Path]:
    storyboard = _load_storyboard(output_dir)
    scenes = storyboard["scenes"]
    scenes_dir = output_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scene in scenes:
        index = int(scene["index"])
        dest = scenes_dir / f"{index:04d}.png"
        if dest.exists() and dest.stat().st_size >= 1000:
            written.append(dest)
            continue
        cond, cond_note = _scene_condition(output_dir, scene.get("characters", []), index)
        prompt = scene["prompt"]
        if cond_note:
            prompt = f"{cond_note}Scene: {prompt}"
        prompt = f"{prompt}{RESTRAINT}"
        print(f"  scene {index + 1}/{len(scenes)} (chars: {','.join(scene.get('characters', [])) or 'none'}, chain: {'yes' if index > 0 else 'first'})")
        # Panel guard: qwen-image sometimes reproduces the reference sheet as
        # side-by-side panels. Detect and resubmit with a fresh seed.
        for attempt in range(3):
            qwen_image_to_file(
                prompt,
                dest,
                SCENE_WIDTH,
                SCENE_HEIGHT,
                seed=2000 + index + attempt * 100,
                ref_image=cond,
                why=f"scene {index} {scene.get('text_excerpt', '')[:60]}",
                negative_prompt=_forbidden_negative(storyboard) or None,
            )
            borders = panel_borders(dest)
            if not borders:
                break
            log.warning("scene %04d panelized (borders at %s, attempt %d) — resubmitting", index, borders, attempt + 1)
            print(f"    scene {index + 1}: panelized (borders {borders}) — retry {attempt + 1}/3")
            if attempt < 2:
                dest.unlink(missing_ok=True)
        written.append(dest)
    return written


__all__ = [
    "build_contact_sheet",
    "generate_character_refs",
    "generate_scene_images",
    "qwen_image_to_file",
]
