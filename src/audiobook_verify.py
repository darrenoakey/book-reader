"""Strict validation of a produced audiobook ``.m4b`` against its assembly manifest.

Assembling is ``m4b_assemble``'s job; this module only reads. It proves, with the
real ``ffmpeg``/``ffprobe`` binaries, that a delivered file is the book the
assembly step recorded:

  * the container probes, every audio packet decodes, and the decode reaches the
    full container duration (a truncated encode fails);
  * the file bytes hash to the ``m4b_sha256`` the manifest recorded;
  * chapter markers match the manifest in count, order, titles and start/end
    times, where the manifest itself is derived from the chapter recordings by
    :func:`expectation_from_audio` using ``m4b_assemble``'s own arithmetic — so
    it is evidence, never a hand-typed claim;
  * ``title``/``album``/``artist`` equal the JOB's title and author, which the
    caller passes in. The manifest is a mutable file and is not trusted for
    identity;
  * the attached cover exists, is a square, and is BYTE-identical to the
    generated art. ``m4b_assemble`` embeds it with ``-c:v copy``, which was
    measured to preserve the JPEG bitstream exactly, so byte parity is owed and
    any difference is a defect rather than a re-encode to be tolerated.

Usage::

    from src.audiobook_verify import expectation_from_audio, verify_audiobook, write_manifest

    write_manifest(project / MANIFEST_NAME,
                   expectation_from_audio(project / "audio", title=title, author=author))
    report = verify_audiobook(project / f"{project.name}.m4b", project / MANIFEST_NAME,
                              title=title, author=author)
    report.raise_for_status()      # ValueError listing every problem
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

MANIFEST_NAME = "audiobook_assembly.json"
SCHEMA = 1

# AAC frame rounding accumulates across chapters; one second plus a hair per
# hour is generous without letting a missing chapter hide.
DURATION_TOLERANCE_SEC = 1.0
DURATION_TOLERANCE_RATIO = 0.001
CHAPTER_TIME_TOLERANCE_SEC = 0.05
COVER_MIN_SIDE = 300
MAX_ERROR_LINES = 20


# ##################################################################
# hashing
# byte and canonical-json digests behind every piece of evidence
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


# ##################################################################
# probe m4b
# one ffprobe pass over format, streams and chapters; never raises on a bad file
def probe_m4b(path: Path, *, timeout: float = 120.0) -> dict[str, Any]:
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "exit": None,
            "stderr": f"ffprobe timed out after {timeout:.0f}s",
            "info": {},
        }
    if result.returncode != 0:
        return {"exit": result.returncode, "stderr": result.stderr.strip(), "info": {}}
    try:
        return {"exit": 0, "stderr": "", "info": json.loads(result.stdout or "{}")}
    except json.JSONDecodeError as exc:
        return {
            "exit": result.returncode,
            "stderr": f"unparsable ffprobe json: {exc}",
            "info": {},
        }


# ##################################################################
# decode audio
# decode every audio packet to the null muxer and record how far it got
def decode_audio(path: Path, *, timeout: float = 1800.0) -> dict[str, Any]:
    cmd = [
        FFMPEG,
        "-nostdin",
        "-v",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "exit": None,
            "errors": [f"decode timed out after {timeout:.0f}s"],
            "decoded_seconds": None,
        }
    decoded: float | None = None
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "out_time_us" and value.strip().isdigit():
            decoded = int(value.strip()) / 1_000_000
    errors = [line for line in result.stderr.splitlines() if line.strip()]
    return {
        "exit": result.returncode,
        "errors": errors[:MAX_ERROR_LINES],
        "error_count": len(errors),
        "decoded_seconds": decoded,
    }


# ##################################################################
# extract cover
# copy the attached picture's packets out verbatim: the exact embedded bitstream
def extract_cover(path: Path, dest: Path, *, timeout: float = 120.0) -> Path | None:
    cmd = [
        FFMPEG,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "data",
        str(dest),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        return None
    return dest


# ##################################################################
# image size
# (width, height) of a decodable image, or None
def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as opened:
            opened.load()
            return opened.size
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


# ##################################################################
# ms truncation
# m4b_assemble writes TIMEBASE=1/1000 integers, so the file carries truncated times
def _ms(seconds: float) -> float:
    return math.floor(seconds * 1000 + 1e-9) / 1000


# ##################################################################
# chapter expectation
# one chapter marker the assembled file must reproduce
@dataclass(frozen=True)
class ChapterExpectation:
    title: str
    start: float
    end: float
    source: str | None = None

    def as_dict(self, index: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": index,
            "title": self.title,
            "start": self.start,
            "end": self.end,
        }
        if self.source is not None:
            payload["source"] = self.source
        return payload


# ##################################################################
# assembly expectation
# what the assembled M4B must be: duration, chapter markers, cover bytes
@dataclass(frozen=True)
class AssemblyExpectation:
    duration_seconds: float
    chapters: tuple[ChapterExpectation, ...]
    cover_name: str | None = None
    cover_sha256: str | None = None
    announcements: bool = False
    album: str | None = None

    @property
    def chapter_titles(self) -> tuple[str, ...]:
        return tuple(chapter.title for chapter in self.chapters)

    def as_manifest(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "version": SCHEMA,
            "album": self.album,
            "duration_seconds": round(self.duration_seconds, 6),
            "announcements": self.announcements,
            "cover_name": self.cover_name,
            "cover_sha256": self.cover_sha256,
            "chapters": [chapter.as_dict(index) for index, chapter in enumerate(self.chapters)],
        }
        return {**body, "fingerprint": canonical_digest(body)}


# ##################################################################
# expectation from audio
# reproduce m4b_assemble's timing arithmetic from the chapter recordings alone
def expectation_from_audio(
    audio_dir: Path,
    *,
    album: str | None = None,
    title: str = "",
    author: str = "",
    cover: Path | None = None,
    max_chapters: int = 0,
    timeout: float = 120.0,
) -> AssemblyExpectation:
    from src.m4b_assemble import (
        PAUSE_AFTER_ANNOUNCEMENT_SEC,
        PAUSE_AFTER_CHIME_SEC,
        PAUSE_BETWEEN_CHAPTERS_SEC,
        find_cover_image,
        number_duplicate_titles,
    )

    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise ValueError(f"audio directory not found: {audio_dir}")
    chapter_files = sorted(f for f in audio_dir.glob("*.wav") if not f.stem.endswith(".announce") and f.stem != "chime")
    if max_chapters > 0:
        chapter_files = chapter_files[:max_chapters]
    if not chapter_files:
        raise ValueError(f"no chapter audio files in {audio_dir}")

    chime_path = audio_dir / "chime.wav"
    chime_seconds = _audio_seconds(chime_path, timeout=timeout) if chime_path.is_file() else None
    markers: list[dict[str, Any]] = []
    current = 0.0
    announced = False
    for chapter_file in chapter_files:
        announce_path = audio_dir / f"{chapter_file.stem}.announce.wav"
        start = current
        if chime_seconds is not None and announce_path.is_file():
            announced = True
            current += chime_seconds + PAUSE_AFTER_CHIME_SEC
            current += _audio_seconds(announce_path, timeout=timeout) + PAUSE_AFTER_ANNOUNCEMENT_SEC
        duration = _audio_seconds(chapter_file, timeout=timeout)
        markers.append(
            {
                "title": chapter_file.stem.split("-", 1)[-1].replace("_", " ").title(),
                "start": start,
                "end": current + duration,
                "_start_ms": _ms(start),
                "_end_ms": _ms(current + duration),
                "source": chapter_file.name,
            }
        )
        current += duration + PAUSE_BETWEEN_CHAPTERS_SEC
    number_duplicate_titles(markers)

    covered = Path(cover) if cover is not None else find_cover_image(audio_dir.parent)
    return AssemblyExpectation(
        duration_seconds=current,
        chapters=tuple(
            ChapterExpectation(
                title=marker["title"],
                start=marker["_start_ms"],
                end=marker["_end_ms"],
                source=marker["source"],
            )
            for marker in markers
        ),
        cover_name=covered.name if covered is not None else None,
        cover_sha256=sha256_file(covered) if covered is not None else None,
        announcements=announced,
        album=album if album is not None else (title or None),
    )


def _audio_seconds(path: Path, *, timeout: float = 120.0) -> float:
    cmd = [
        FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe gave no duration for {path}: {result.stdout!r}") from exc


# ##################################################################
# manifest round trip
# the assembly manifest is the contract a later verification runs against
def write_manifest(path: Path, expectation: AssemblyExpectation, *, m4b: Path | None = None) -> Path:
    path = Path(path)
    manifest = expectation.as_manifest()
    if m4b is not None:
        manifest["m4b_sha256"] = sha256_file(m4b)
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    manifest["fingerprint"] = canonical_digest(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)
    return path


def manifest_is_intact(manifest: Any) -> bool:
    """True when *manifest* carries a fingerprint matching its own body.

    This is the non-raising form of the check :func:`load_manifest` enforces, for
    callers that want to reuse a previous assembly only if its record was not
    edited, and to fall back to a rebuild otherwise::

        if manifest_is_intact(json.loads(manifest_path.read_text())):
            return existing_m4b
    """
    if not isinstance(manifest, Mapping):
        return False
    if manifest.get("version") != SCHEMA:
        return False
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return False
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    try:
        return canonical_digest(body) == fingerprint
    except (TypeError, ValueError):
        return False


def load_manifest(path: Path) -> tuple[AssemblyExpectation, dict[str, Any]]:
    """Read an assembly manifest, refusing one whose own fingerprint does not match."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"{path.name} is not an assembly manifest object")
    for required in ("version", "duration_seconds", "chapters", "fingerprint"):
        if required not in manifest:
            raise ValueError(f"{path.name} has no {required!r} field")
    if manifest["version"] != SCHEMA:
        raise ValueError(f"{path.name} is version {manifest['version']!r}, this verifier reads version {SCHEMA}")
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if manifest["fingerprint"] != canonical_digest(body):
        raise ValueError(f"{path.name} failed its own fingerprint; the manifest was edited")
    chapters = tuple(
        ChapterExpectation(
            title=str(chapter["title"]),
            start=float(chapter["start"]),
            end=float(chapter["end"]),
            source=str(chapter["source"]) if chapter.get("source") else None,
        )
        for chapter in manifest["chapters"]
    )
    return (
        AssemblyExpectation(
            duration_seconds=float(manifest["duration_seconds"]),
            chapters=chapters,
            cover_name=manifest.get("cover_name"),
            cover_sha256=manifest.get("cover_sha256"),
            announcements=bool(manifest.get("announcements")),
            album=manifest.get("album"),
        ),
        manifest,
    )


# ##################################################################
# verification report
# every named check with its measured detail, plus a digest of the stable proof
@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    problems: tuple[str, ...]
    checks: dict[str, dict[str, Any]]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "checks": self.checks,
            "fingerprint": self.fingerprint,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ValueError(f"audiobook verification failed: {'; '.join(self.problems)}")


# ##################################################################
# verify audiobook
# the one entry point: real decode, manifest parity, job identity, exact cover
def verify_audiobook(
    m4b: Path,
    manifest: Path,
    *,
    title: str,
    author: str,
    cover: Path | None = None,
    duration_tolerance: float = DURATION_TOLERANCE_SEC,
    chapter_tolerance: float = CHAPTER_TIME_TOLERANCE_SEC,
    probe_timeout: float = 120.0,
    decode_timeout: float = 1800.0,
) -> VerificationReport:
    """Prove ``m4b`` is the audiobook the manifest describes for *this* job.

    ``title``/``author`` come from the job, not the manifest: a mutable file is
    not trusted to name the book. ``cover`` is the generated art the assembly
    embedded; it defaults to the file named by the manifest beside it.
    """
    m4b, manifest_path = Path(m4b), Path(manifest)
    expected, recorded = load_manifest(manifest_path)
    art = (
        Path(cover)
        if cover is not None
        else (manifest_path.parent / expected.cover_name if expected.cover_name else None)
    )
    checks: dict[str, tuple[bool, str]] = {}

    def record(name: str, ok: bool, detail: str) -> bool:
        checks[name] = (ok, detail)
        return ok

    evidence: dict[str, Any] = {}

    def finish() -> VerificationReport:
        problems = tuple(f"{name}: {detail}" for name, (ok, detail) in checks.items() if not ok)
        stable = {name: {"ok": ok, "detail": detail} for name, (ok, detail) in checks.items()}
        return VerificationReport(
            ok=not problems,
            problems=problems,
            checks=stable,
            fingerprint=canonical_digest({"checks": stable, "evidence": evidence}),
        )

    # ------------------------------------------------------------- readable
    if not m4b.is_file() or m4b.stat().st_size == 0:
        record("m4b_readable", False, f"{m4b} is missing or empty")
        return finish()
    m4b_sha = sha256_file(m4b)
    evidence["m4b_sha256"] = m4b_sha
    if not record("m4b_readable", True, f"read {m4b.name} ({m4b.stat().st_size} bytes)"):
        return finish()

    recorded_sha = recorded.get("m4b_sha256")
    if recorded_sha is None:
        record(
            "m4b_bytes_bound",
            False,
            "the manifest records no m4b_sha256; the delivered bytes are unbound",
        )
    else:
        record(
            "m4b_bytes_bound",
            recorded_sha == m4b_sha,
            f"sha256 {m4b_sha}"
            if recorded_sha == m4b_sha
            else f"sha256 {m4b_sha} but the manifest records {recorded_sha}",
        )

    probed = probe_m4b(m4b, timeout=probe_timeout)
    if not record(
        "m4b_probes",
        probed["exit"] == 0 and bool(probed["info"]),
        "ffprobe read the container"
        if probed["exit"] == 0 and probed["info"]
        else f"ffprobe failed: {probed['stderr'][:300]}",
    ):
        return finish()
    info = probed["info"]

    # ------------------------------------------------------------- decode
    decoded = decode_audio(m4b, timeout=decode_timeout)
    evidence["decoded_seconds"] = decoded["decoded_seconds"]
    record(
        "decode_clean",
        decoded["exit"] == 0 and decoded["error_count"] == 0,
        "decoded every audio packet cleanly"
        if decoded["exit"] == 0 and decoded["error_count"] == 0
        else f"ffmpeg exit {decoded['exit']}, {decoded['error_count']} error(s): {'; '.join(decoded['errors'][:3])}",
    )

    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    container_seconds = float(fmt.get("duration") or 0)
    tags = {str(key).lower(): value for key, value in (fmt.get("tags") or {}).items()}
    evidence["duration_seconds"] = round(container_seconds, 3)

    allowed = max(duration_tolerance, container_seconds * DURATION_TOLERANCE_RATIO)
    if decoded["decoded_seconds"] is None:
        record(
            "decoded_duration",
            False,
            "the decode pass reported no out_time; no proof of complete decoding",
        )
    else:
        record(
            "decoded_duration",
            abs(decoded["decoded_seconds"] - container_seconds) <= allowed,
            f"decoded {decoded['decoded_seconds']:.3f}s of a {container_seconds:.3f}s container",
        )

    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    record(
        "single_audio_stream",
        len(audios) == 1,
        f"{len(audios)} audio streams, expected exactly 1",
    )

    # ------------------------------------------------------------- duration
    delta = container_seconds - expected.duration_seconds
    record(
        "duration_parity",
        abs(delta) <= allowed,
        f"{container_seconds:.3f}s vs manifest {expected.duration_seconds:.3f}s (delta {delta:+.3f}s, allowed {allowed:.3f}s)",
    )

    # ------------------------------------------------------------- chapters
    chapters = info.get("chapters") or []
    titles = [str((chapter.get("tags") or {}).get("title") or "") for chapter in chapters]
    starts = [float(chapter.get("start_time") or 0) for chapter in chapters]
    ends = [float(chapter.get("end_time") or 0) for chapter in chapters]
    evidence["chapter_titles"] = titles
    record(
        "chapter_count_parity",
        len(chapters) == len(expected.chapters),
        f"{len(chapters)} markers, manifest says {len(expected.chapters)}",
    )
    record(
        "chapter_order_parity",
        *_chapter_problem(titles, starts, ends, expected, chapter_tolerance),
    )

    # ------------------------------------------------------------- identity
    record(
        "title_tag",
        tags.get("title") == title,
        f"title={tags.get('title')!r}, job says {title!r}",
    )
    record(
        "album_tag",
        tags.get("album") == title,
        f"album={tags.get('album')!r}, job title is {title!r}",
    )
    record(
        "artist_tag",
        tags.get("artist") == author,
        f"artist={tags.get('artist')!r}, job author is {author!r}",
    )

    # ------------------------------------------------------------- cover
    covers = [stream for stream in streams if (stream.get("disposition") or {}).get("attached_pic") == 1]
    if not record(
        "cover_stream",
        len(covers) == 1,
        f"{len(covers)} attached pictures, expected exactly 1",
    ):
        return finish()
    cover_stream = covers[0]
    evidence["cover_stream"] = {
        "codec_name": cover_stream.get("codec_name"),
        "width": cover_stream.get("width"),
        "height": cover_stream.get("height"),
    }

    if art is None or not art.is_file():
        record(
            "cover_source_parity",
            False,
            f"the generated art is unavailable: {art or expected.cover_name!r}",
        )
        return finish()
    with tempfile.TemporaryDirectory(prefix="audiobook-verify-") as scratch:
        embedded = extract_cover(m4b, Path(scratch) / "embedded_cover.jpg")
        if not record(
            "cover_decodable",
            embedded is not None,
            "extracted the attached picture" if embedded else "could not extract the attached picture",
        ):
            return finish()
        size = image_size(embedded)
        if not record(
            "cover_decodable",
            size is not None,
            f"attached picture decodes to {size}" if size else "the attached picture does not decode",
        ):
            return finish()
        record(
            "cover_geometry",
            size[0] == size[1] and size[0] >= COVER_MIN_SIDE,
            f"{size[0]}x{size[1]}, expected a square of at least {COVER_MIN_SIDE}px",
        )
        embedded_sha, art_sha = sha256_file(embedded), sha256_file(art)
        evidence["cover_sha256"] = {"embedded": embedded_sha, "art": art_sha}
        record(
            "cover_source_parity",
            embedded_sha == art_sha,
            f"embedded cover is byte-identical to {art.name}"
            if embedded_sha == art_sha
            else f"embedded cover sha256 {embedded_sha} is not {art.name} {art_sha}; -c:v copy must preserve the art bytes",
        )
        if expected.cover_sha256 is not None:
            record(
                "cover_bound_to_manifest",
                expected.cover_sha256 == art_sha,
                f"{art.name} sha256 {art_sha} matches the manifest"
                if expected.cover_sha256 == art_sha
                else f"{art.name} sha256 {art_sha} is not the manifest's {expected.cover_sha256}",
            )
    return finish()


# ##################################################################
# chapter problem
# first chapter that leaves the manifest in title, order, start or end
def _chapter_problem(
    titles: list[str],
    starts: list[float],
    ends: list[float],
    expected: AssemblyExpectation,
    tolerance: float,
) -> tuple[bool, str]:
    if len(titles) != len(expected.chapters):
        return (
            False,
            f"{len(titles)} markers cannot match {len(expected.chapters)} manifest chapters",
        )
    for index, chapter in enumerate(expected.chapters):
        if titles[index] != chapter.title:
            return (
                False,
                f"chapter {index} is {titles[index]!r}, manifest says {chapter.title!r}",
            )
        if abs(starts[index] - chapter.start) > tolerance:
            return (
                False,
                f"chapter {index} ({chapter.title!r}) starts at {starts[index]:.3f}s, manifest says {chapter.start:.3f}s",
            )
        # the mp4 muxer closes the pause up to the next marker, so an end may be
        # extended that far but never shortened
        ceiling = expected.chapters[index + 1].start if index + 1 < len(expected.chapters) else chapter.end
        if ends[index] < chapter.end - tolerance or ends[index] > max(chapter.end, ceiling) + tolerance:
            return False, (
                f"chapter {index} ({chapter.title!r}) spans {starts[index]:.3f}-{ends[index]:.3f}s, "
                f"manifest says {chapter.start:.3f}-{chapter.end:.3f}s"
            )
    return True, f"{len(titles)} chapters match the manifest in order and timing"


__all__ = [
    "MANIFEST_NAME",
    "SCHEMA",
    "AssemblyExpectation",
    "ChapterExpectation",
    "VerificationReport",
    "canonical_digest",
    "decode_audio",
    "expectation_from_audio",
    "extract_cover",
    "image_size",
    "load_manifest",
    "manifest_is_intact",
    "probe_m4b",
    "sha256_file",
    "verify_audiobook",
    "write_manifest",
]
