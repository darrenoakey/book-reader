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

RESTRAINT = " No text, no captions, no watermark, no extra limbs, anatomically correct."

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
            if "failed" in msg or "cancelled" in msg or "timed out" in msg:
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
    if ref_image is not None:
        clean["image_file"] = stage_file(ref_image)
    reason = why or f"image {dest.name}"
    jid = _submit(client, "qwen-image", clean, why=reason)
    _fetch_image(client, jid, clean, dest, reason)
    return dest


# ##################################################################
# build contact sheet
# compose labelled portraits side by side: n tiles of tile×tile plus a
# LABEL_BAND strip underneath with each character's display name
def build_contact_sheet(items: list[tuple[str, Path]], dest: Path, tile: int = REF_SIZE // 2) -> Path:
    if not items:
        raise ValueError("contact sheet needs at least one portrait")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (tile * len(items), tile + LABEL_BAND), (18, 18, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (name, path) in enumerate(items):
        portrait = Image.open(path).convert("RGB").resize((tile, tile), Image.LANCZOS)
        sheet.paste(portrait, (i * tile, 0))
        label = name.replace("-", " ").title()
        bbox = draw.textbbox((0, 0), label)
        x = i * tile + (tile - (bbox[2] - bbox[0])) // 2
        draw.text((x, tile + (LABEL_BAND - (bbox[3] - bbox[1])) // 2), label, fill=(235, 235, 235))
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
            f"Facing camera, full face visible, neutral dark background, face large and unmistakable. "
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
def _scene_ref(output_dir: Path, characters: list[str], index: int) -> tuple[Path | None, str]:
    refs_dir = output_dir / "refs"
    available = [(cid, refs_dir / f"{cid}.png") for cid in characters if (refs_dir / f"{cid}.png").exists()]
    if not available:
        return None, ""
    if len(available) == 1:
        cid, path = available[0]
        return path, f"The reference image is a portrait of {cid.replace('-', ' ')}; keep this exact face, hair, and clothing."
    sheet = build_contact_sheet(available, output_dir / "scenes" / ".sheets" / f"{index:04d}.png")
    names = ", ".join(cid.replace("-", " ") for cid, _ in available)
    return sheet, (
        f"The reference image is a labelled strip of character portraits ({names}); "
        "use these exact faces, hair, and clothing for the matching characters in the scene."
    )


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
        ref, ref_note = _scene_ref(output_dir, scene.get("characters", []), index)
        prompt = scene["prompt"]
        if ref_note:
            prompt = f"{ref_note} Scene: {prompt}"
        prompt = f"{prompt}{RESTRAINT}"
        print(f"  scene {index + 1}/{len(scenes)} (chars: {','.join(scene.get('characters', [])) or 'none'})")
        qwen_image_to_file(
            prompt,
            dest,
            SCENE_WIDTH,
            SCENE_HEIGHT,
            seed=2000 + index,
            ref_image=ref,
            why=f"scene {index} {scene.get('text_excerpt', '')[:60]}",
        )
        written.append(dest)
    return written


__all__ = [
    "build_contact_sheet",
    "generate_character_refs",
    "generate_scene_images",
    "qwen_image_to_file",
]
