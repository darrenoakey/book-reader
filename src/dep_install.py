"""Release-gate dependency install.

A cold `pip install` of this tree downloads the world and dominated a 404s
release. The gate worktree is persistent, and uv's content-addressed cache
already holds those wheels after one resolve, so a missing virtualenv is a
local install. An unchanged requirements fingerprint skips install entirely.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

STAMP_NAME = ".install-stamp"
_SKIP_DIRS = {".git", "__pycache__", ".venv", "dist", "build", ".mypy_cache", ".pytest_cache"}


# ##################################################################
# find uv
# The release host keeps uv off the minimal gate PATH; look next to the
# usual user install before falling back to a network pip install.
def find_uv() -> Path | None:
    found = shutil.which("uv")
    if found:
        return Path(found)
    for candidate in (
        Path.home() / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


# ##################################################################
# local dependency files
# Hash the real file:// tree so an arbiter-client edit invalidates the stamp
# without walking another project's virtualenv or build debris.
def _local_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS and not name.endswith(".egg-info")]
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            files.append(Path(dirpath) / name)
    return sorted(files)


# ##################################################################
# dependency fingerprint
# requirements.txt plus every file:// dependency it names. Missing deps fail
# closed so a stale stamp cannot skip a broken install.
def dependency_fingerprint(root: Path) -> str:
    requirements = root / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes())
    for line in requirements.read_text(encoding="utf-8").splitlines():
        bare = line.split("#", 1)[0].strip()
        if "file://" not in bare:
            continue
        raw = bare.split("file://", 1)[1].strip()
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"requirements file dependency does not exist: {path}")
        for dep_file in _local_files(path):
            digest.update(dep_file.relative_to(path).as_posix().encode())
            digest.update(dep_file.read_bytes())
    return digest.hexdigest()


# ##################################################################
# dependencies current
# True only when this venv recorded the current fingerprint after a successful install.
def dependencies_current(venv: Path, fingerprint: str) -> bool:
    python = venv / "bin" / "python"
    stamp = venv / STAMP_NAME
    if not python.is_file() or not stamp.is_file():
        return False
    return stamp.read_text(encoding="utf-8").strip() == fingerprint


# ##################################################################
# write stamp
# Record the fingerprint only after install returns success.
def write_stamp(venv: Path, fingerprint: str) -> None:
    (venv / STAMP_NAME).write_text(fingerprint + "\n", encoding="utf-8")


# ##################################################################
# install requirements
# Prefer an offline uv install from the local cache. A cache miss falls back
# to one online uv resolve, then pip, so a fresh host still installs.
def install_requirements(root: Path, python: Path) -> int:
    requirements = root / "requirements.txt"
    uv = find_uv()
    if uv is not None:
        # copy: the uv cache and the gate worktree are on different volumes, so
        # hardlinks fail and the fallback copy is the only mode that works.
        base = [str(uv), "pip", "install", "--python", str(python), "--link-mode=copy"]
        print("Installing dependencies from local cache...", flush=True)
        if subprocess.call([*base, "--offline", "-r", str(requirements)], cwd=root) == 0:
            return 0
        print("Local cache miss; installing dependencies...", flush=True)
        if subprocess.call([*base, "-r", str(requirements)], cwd=root) == 0:
            return 0
    pip = python.parent / "pip"
    print("Installing dependencies with pip...", flush=True)
    return subprocess.call([str(pip), "install", "-r", str(requirements)], cwd=root)


# ##################################################################
# ensure installed
# Create nothing itself: the caller owns the virtualenv. Skip the installer
# when the stamp matches so a warm gate does not rebuild path dependencies.
def ensure_installed(root: Path, venv: Path, python: Path) -> int:
    fingerprint = dependency_fingerprint(root)
    if dependencies_current(venv, fingerprint):
        print("Dependencies already installed", flush=True)
        return 0
    if not python.is_file():
        print(f"virtualenv python missing: {python}", file=sys.stderr, flush=True)
        return 1
    rc = install_requirements(root, python)
    if rc == 0:
        write_stamp(venv, fingerprint)
        print("Dependencies installed successfully", flush=True)
    return rc
