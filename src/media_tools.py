"""Absolute resolution of the external media binaries the pipeline shells out to.

A detached audiobook worker inherits whatever PATH its launching process had,
not the operator's interactive shell PATH: a worker started from a minimal
launchd/agent environment has no Homebrew directory, so a bare ``"ffmpeg"``
raised ``FileNotFoundError`` in the middle of a multi-hour render (Dragonsinger,
after 14 hours of synthesis, at chapter concatenation).

Every media call therefore resolves the binary once through :func:`ffmpeg` /
:func:`ffprobe`: the ambient PATH first (so an operator override still wins),
then the documented per-platform install locations. Resolution is cached per
process and fails closed with one actionable error instead of an ENOENT deep
inside subprocess.
"""

from __future__ import annotations

import functools
import os
import shutil
from pathlib import Path

FFMPEG_NAME = "ffmpeg"
FFPROBE_NAME = "ffprobe"

# Fixed install locations, in preference order. Homebrew (Apple silicon and
# Intel), MacPorts, the Linux package managers, and a static build dropped on
# the PATH-only system directories.
_INSTALL_DIRS = (
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path("/opt/local/bin"),
    Path("/home/linuxbrew/.linuxbrew/bin"),
    Path("/usr/bin"),
    Path("/bin"),
)
_EXTRA_CANDIDATES = (
    Path.home() / ".local" / "bin",
    Path.home() / "bin",
)


class MediaToolMissing(RuntimeError):
    """A required media binary is not installed where this host can find it."""


# ##################################################################
# resolve one binary
# ambient PATH first (explicit operator intent), then the known install dirs;
# no shell interpolation and no bare name that would ENOENT later.
@functools.cache
def resolve(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    candidates = [directory / name for directory in _INSTALL_DIRS]
    candidates += [directory / name for directory in _EXTRA_CANDIDATES]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise MediaToolMissing(
        f"{name} was not found on PATH ({os.environ.get('PATH', '')}) or in any known install "
        "directory; install it (macOS: `brew install ffmpeg`) or put it on PATH"
    )


def ffmpeg() -> str:
    return resolve(FFMPEG_NAME)


def ffprobe() -> str:
    return resolve(FFPROBE_NAME)


def describe() -> dict[str, str | None]:
    """Non-fatal inventory for diagnostics: which binary would be used, if any."""
    inventory: dict[str, str | None] = {}
    for name in (FFMPEG_NAME, FFPROBE_NAME):
        try:
            inventory[name] = resolve(name)
        except MediaToolMissing:
            inventory[name] = None
    return inventory


# ##################################################################
# interpreter dirs for a spawned child
# a worker re-exec'd by an already-running pipeline must not lose a resolution
# that only this process could make, so the resolved binary's directory is
# prepended for children the pipeline starts itself.
def child_path() -> str:
    path = os.environ.get("PATH", "")
    try:
        directory = str(Path(ffmpeg()).parent)
    except MediaToolMissing:
        return path
    parts = [part for part in path.split(os.pathsep) if part]
    if directory in parts:
        return path
    return os.pathsep.join([directory, *parts])


__all__ = [
    "FFMPEG_NAME",
    "FFPROBE_NAME",
    "MediaToolMissing",
    "child_path",
    "describe",
    "ffmpeg",
    "ffprobe",
    "resolve",
]
