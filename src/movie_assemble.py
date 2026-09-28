"""Movie assembly: turn the storyboard scene stills + chapter audio into a
Ken Burns "audiobook movie" — every still slowly pans or zooms so the picture
is never static, cut exactly on storyboard scene boundaries, muxed with the
full narration track.

Frame-exactness is the core design constraint (A/V sync over a long film):

  * the storyboard's scene times are seconds against the CONCATENATION of the
    chapter wavs (see movie_storyboard.load_global_lines), so the movie's
    audio track is exactly that concatenation, in the same order;
  * each scene renders at FPS with ``frames_i = round(end*FPS) -
    round(start*FPS)`` frames — cumulative rounding can never drift;
  * the final scene absorbs any remainder so total frames ==
    round(audio_seconds*FPS) exactly;
  * no ``-shortest`` anywhere (it silently drops tail frames); the mux caps
    nothing — video and audio are independently exact.

Segments render once each (idempotent skip) so a rerender after replacing a
scene image only re-renders that segment and re-concats.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from src.audio_synth import concat_wavs, wav_duration
from src.title_page import TITLE_SECONDS

FPS = 30
WIDTH = 1920
HEIGHT = 1080

# Camera grammar: every move is ONE linear transition from a START view to an
# END view, where a view is (zoom, cx, cy) — zoom factor and crop-centre as
# frame fractions. A scene gets exactly ONE movement: constant velocity from
# start to end across the whole segment, never a change of direction. A move
# may combine a zoom and a pan (still one movement). STATICS hold perfectly
# still — every STATIC_EVERY-th scene rests.
View = tuple[float, float, float]

MOVES: dict[str, tuple[View, View]] = {
    "zoom-in": ((1.00, 0.50, 0.50), (1.14, 0.50, 0.50)),
    "zoom-out": ((1.14, 0.50, 0.50), (1.00, 0.50, 0.50)),
    "drift-right": ((1.14, 0.34, 0.50), (1.14, 0.66, 0.50)),
    "drift-left": ((1.14, 0.66, 0.50), (1.14, 0.34, 0.50)),
    "rise": ((1.06, 0.50, 0.64), (1.18, 0.50, 0.42)),
    "settle": ((1.18, 0.50, 0.40), (1.06, 0.50, 0.62)),
    "climb": ((1.14, 0.50, 0.66), (1.14, 0.50, 0.36)),
}
STATICS: dict[str, tuple[View, View]] = {
    "still-centre": ((1.12, 0.50, 0.50), (1.12, 0.50, 0.50)),
    "still-left": ((1.14, 0.40, 0.48), (1.14, 0.40, 0.48)),
    "still-right": ((1.14, 0.60, 0.52), (1.14, 0.60, 0.52)),
}
_MOVE_CYCLE = tuple(MOVES)
_STATIC_CYCLE = tuple(STATICS)
STATIC_EVERY = 5  # every 5th scene holds still

# Anti-jitter: zoompan steps in integer source pixels, so pans judder when a
# frame advances <1 source px. Pre-scaling the still 4x (once, cached — not
# in the per-frame filter graph) makes each step 0.25 output px: invisible.
_SUPER = 4


# ##################################################################
# scene move
# the move for scene `index`: every STATIC_EVERY-th scene rests; the rest
# cycle the moving moves (offset by one when a title card took zoom-in)
def scene_move(index: int, title: bool = False) -> str:
    if index % STATIC_EVERY == STATIC_EVERY - 1:
        return _STATIC_CYCLE[(index // STATIC_EVERY) % len(_STATIC_CYCLE)]
    ordinal = index - (index // STATIC_EVERY)  # count of moving scenes so far
    return _MOVE_CYCLE[(ordinal + (1 if title else 0)) % len(_MOVE_CYCLE)]


# ##################################################################
# run checked
# ffmpeg/ffprobe wrapper that raises with stderr on failure
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed: {result.stderr[-2000:]}")
    return result


# ##################################################################
# probe duration
# container duration in seconds via ffprobe
def probe_duration(path: Path) -> float:
    out = _run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
    ).stdout
    return float(out.strip())


# ##################################################################
# probe frames
# exact decoded frame count via ffprobe (metadata frame counts lie)
def probe_frames(path: Path) -> int:
    out = _run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
        ]
    ).stdout
    return int(out.strip())


# ##################################################################
# zoompan filter
# one camera move as a complete -vf chain: a single linear ramp from the
# start view to the end view (constant velocity, one direction), delivering
# WIDTHxHEIGHT yuv420p at FPS with exactly `frames` frames per input frame.
# Input MUST be pre-scaled (see _prescaled) — there is no in-graph upscale.
def _lerp(a: float, b: float, t: str) -> str:
    if a == b:
        return f"{a:g}"
    return f"{a:g}+({b - a:g})*{t}"


def zoompan_filter(move: str, frames: int) -> str:
    if frames < 1:
        raise ValueError("frames must be >= 1")
    views = MOVES.get(move) or STATICS.get(move)
    if views is None:
        raise ValueError(f"unknown move {move!r}")
    (z0, cx0, cy0), (z1, cx1, cy1) = views
    t = f"on/{frames - 1}" if frames > 1 else "1"  # 0..1 progress over the segment
    z = _lerp(z0, z1, t)
    x = f"(iw-iw/zoom)*({_lerp(cx0, cx1, t)})"
    y = f"(ih-ih/zoom)*({_lerp(cy0, cy1, t)})"
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},format=yuv420p"


# ##################################################################
# prescaled
# the still upscaled _SUPERx, cached beside the segments (rebuilt if the
# source image is newer) — one ffmpeg call per segment, not per frame
def _prescaled(image: Path, cache_dir: Path) -> Path:
    scaled = cache_dir / f"{image.stem}-{_SUPER}x.png"
    if not scaled.exists() or scaled.stat().st_mtime < image.stat().st_mtime:
        scaled.parent.mkdir(parents=True, exist_ok=True)
        _run(["ffmpeg", "-y", "-i", str(image), "-vf", f"scale={WIDTH * _SUPER}:-1:flags=lanczos", str(scaled)])
    return scaled


# ##################################################################
# render segment
# render one still into an h264 segment of EXACTLY `frames` frames
def render_segment(image: Path, frames: int, move: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        # A cached segment is only valid for the SAME frame count — when the
        # title card shaves frames off scene 0 its old segment must re-render.
        # (The move is encoded in the filename, so a move-engine change busts
        # the cache automatically.)
        if probe_frames(dest) == frames:
            return dest
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    scaled = _prescaled(image, dest.parent / ".prescaled")
    _run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(scaled),
            "-vf", zoompan_filter(move, frames),
            "-frames:v", str(frames),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-r", str(FPS), "-an", str(dest),
        ]
    )
    actual = probe_frames(dest)
    if actual != frames:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"segment {dest.name} has {actual} frames, expected {frames}")
    return dest


# ##################################################################
# chapter audio order
# the chapter wavs in exactly the order the storyboard offset them
def _chapter_wavs(output_dir: Path) -> list[Path]:
    audio_dir = output_dir / "audio"
    timelines = sorted(audio_dir.glob("*.timeline.json"))
    if not timelines:
        raise ValueError("no audio timelines — run the audio step first")
    wavs = []
    for tl in timelines:
        wav = audio_dir / f"{tl.stem.replace('.timeline', '')}.wav"
        if not wav.exists():
            raise ValueError(f"timeline {tl.name} has no wav {wav.name}")
        wavs.append(wav)
    return wavs


# ##################################################################
# assemble movie
# full step: storyboard scenes + scene stills + chapter audio → movie/movie.mp4
def assemble_movie(output_dir: Path, title: str) -> Path:
    storyboard_path = output_dir / "storyboard.json"
    if not storyboard_path.exists():
        raise ValueError("storyboard.json not found — run the storyboard step first")
    scenes = json.loads(storyboard_path.read_text(encoding="utf-8"))["scenes"]
    if not scenes:
        raise ValueError("storyboard has no scenes")

    movie_dir = output_dir / "movie"
    segments_dir = movie_dir / "segments"
    movie_dir.mkdir(parents=True, exist_ok=True)

    # Audio: the exact concatenation the storyboard timed against.
    audio_full = movie_dir / "audio_full.wav"
    if not audio_full.exists():
        concat_wavs(_chapter_wavs(output_dir), audio_full)
    audio_seconds = wav_duration(audio_full)
    total_frames = round(audio_seconds * FPS)

    # Frame budget per scene: round each boundary, never the durations.
    bounds = [round(float(s["start"]) * FPS) for s in scenes] + [total_frames]

    # Opening title card: when title_page.png exists it holds for the first
    # TITLE_SECONDS, carved out of scene 0's slot (total frames unchanged, so
    # A/V sync is untouched). Scene 0 keeps at least one second of its own.
    title_frames = 0
    title_image = output_dir / "title_page.png"
    if title_image.exists() and title_image.stat().st_size >= 1000:
        first = bounds[1] - bounds[0]
        title_frames = min(round(TITLE_SECONDS * FPS), first - FPS)
        if title_frames < FPS:
            title_frames = 0  # scene 0 too short to share — skip the card

    segments: list[Path] = []
    if title_frames:
        segments.append(render_segment(title_image, title_frames, "zoom-in", segments_dir / "title.zoom-in.mp4"))
        print(f"    title card: {title_frames / FPS:.1f}s")
    for i, scene in enumerate(scenes):
        frames = bounds[i + 1] - bounds[i]
        if i == 0:
            frames -= title_frames
        if frames < 1:
            frames = FPS  # degenerate zero-width scene — give it one second
        image = output_dir / "scenes" / f"{int(scene['index']):04d}.png"
        if not image.exists():
            raise ValueError(f"scene image missing: {image} — run the sceneimages step")
        # One movement per scene, assigned deterministically; every 5th rests.
        move = scene_move(i, title=bool(title_frames))
        segments.append(render_segment(image, frames, move, segments_dir / f"{i:04d}.{move}.mp4"))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(scenes)} segments rendered")

    # Concat segments (all identical codec/geometry → stream copy, no re-encode).
    video_only = movie_dir / "video_only.mp4"
    if not video_only.exists():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")
            list_file = Path(f.name)
        try:
            _run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", str(video_only),
                ]
            )
        finally:
            list_file.unlink(missing_ok=True)
        actual = probe_frames(video_only)
        if actual != total_frames:
            video_only.unlink(missing_ok=True)
            raise RuntimeError(f"concatenated video has {actual} frames, expected {total_frames}")

    # Mux with the narration track. Both sides are independently exact; the
    # container duration may differ by <1 frame of AAC priming, nothing drifts.
    movie_path = movie_dir / "movie.mp4"
    _run(
        [
            "ffmpeg", "-y", "-i", str(video_only), "-i", str(audio_full),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-metadata", f"title={title}",
            str(movie_path),
        ]
    )
    drift = abs(probe_duration(movie_path) - audio_seconds)
    if drift > 0.5:
        raise RuntimeError(f"movie/audio drift {drift:.3f}s exceeds 0.5s tolerance")
    print(f"  movie: {audio_seconds / 60:.1f} min, {total_frames} frames, drift {drift:.3f}s")
    return movie_path


__all__ = ["FPS", "MOVES", "STATICS", "assemble_movie", "probe_duration", "render_segment", "scene_move", "zoompan_filter"]
