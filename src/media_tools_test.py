"""Real resolution tests for the external media binaries.

The production defect these cover: a detached audiobook worker inherited a
PATH with no Homebrew directory, so a bare ``"ffmpeg"`` raised FileNotFoundError
fourteen hours into a render. Every media call now goes through
``src.media_tools``, which must find the binaries on a minimal PATH and fail
closed with one actionable message when they are genuinely absent.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import media_tools
from src.media_tools import MediaToolMissing, child_path, describe, ffmpeg, ffprobe, resolve

MINIMAL_PATHS = ("/usr/bin:/bin", "/nonexistent")


# ##################################################################
# a real ffmpeg path is found with a minimal PATH
# the exact production case: launchd/agent PATH, no Homebrew entry.
@pytest.mark.parametrize("minimal", MINIMAL_PATHS)
def test_resolves_media_binaries_without_the_operators_path(minimal: str) -> None:
    root = str(Path(__file__).resolve().parent.parent)
    script = (
        "import sys\n"
        f"sys.path.insert(0, {root!r})\n"
        "from src.media_tools import ffmpeg, ffprobe\n"
        "print(ffmpeg())\n"
        "print(ffprobe())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": minimal},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    found = result.stdout.split()
    assert len(found) == 2
    for binary in found:
        assert Path(binary).is_file() and os.access(binary, os.X_OK)


# ##################################################################
# the resolved binary really runs
# resolution is only useful if the path executes, so run it for real.
def test_resolved_binaries_execute() -> None:
    version = subprocess.run([ffmpeg(), "-version"], capture_output=True, text=True, check=False, timeout=120)
    assert version.returncode == 0
    assert "ffmpeg version" in version.stdout
    probe = subprocess.run([ffprobe(), "-version"], capture_output=True, text=True, check=False, timeout=120)
    assert probe.returncode == 0


# ##################################################################
# a missing binary fails closed
# one actionable error naming both the PATH and the remedy, never a bare ENOENT.
def test_missing_binary_fails_closed_with_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_tools, "_INSTALL_DIRS", ())
    monkeypatch.setattr(media_tools, "_EXTRA_CANDIDATES", ())
    monkeypatch.setattr(media_tools.shutil, "which", lambda name: None)
    media_tools.resolve.cache_clear()
    try:
        with pytest.raises(MediaToolMissing) as caught:
            media_tools.resolve("ffmpeg")
        message = str(caught.value)
        assert "ffmpeg" in message
        assert "PATH" in message
        assert "brew install ffmpeg" in message
    finally:
        media_tools.resolve.cache_clear()


# ##################################################################
# inventory never raises
# diagnostics report what is absent instead of failing the caller.
def test_describe_reports_absent_binaries_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_tools, "_INSTALL_DIRS", ())
    monkeypatch.setattr(media_tools, "_EXTRA_CANDIDATES", ())
    monkeypatch.setattr(media_tools.shutil, "which", lambda name: None)
    media_tools.resolve.cache_clear()
    try:
        assert describe() == {"ffmpeg": None, "ffprobe": None}
    finally:
        media_tools.resolve.cache_clear()


# ##################################################################
# children inherit the resolved directory
# a worker the pipeline re-exec's must keep the tool this process resolved.
def test_child_path_prepends_the_resolved_directory() -> None:
    path = child_path()
    # the resolved binary's directory must be reachable for the child, whether
    # this process already had it or we had to prepend it
    assert str(Path(ffmpeg()).parent) in path.split(os.pathsep)
    # idempotent: an already-present directory is not duplicated
    previous = os.environ.get("PATH")
    os.environ["PATH"] = path
    try:
        assert child_path() == path
        assert path.split(os.pathsep).count(str(Path(ffmpeg()).parent)) == 1
    finally:
        if previous is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous


# ##################################################################
# a child started from a PATH without the tool still runs the resolved binary
def test_child_path_rescues_a_minimal_parent_path() -> None:
    previous = os.environ.get("PATH")
    os.environ["PATH"] = "/nonexistent"
    try:
        rescued = child_path()
    finally:
        if previous is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous
    assert rescued.split(os.pathsep)[0] == str(Path(ffmpeg()).parent)
    result = subprocess.run(
        [ffmpeg(), "-version"], capture_output=True, text=True, check=False, env={"PATH": rescued}, timeout=120
    )
    assert result.returncode == 0


# ##################################################################
# modules never hold a bare binary name
# a regression here reintroduces the production failure silently.
def test_media_modules_resolve_instead_of_hardcoding_names() -> None:
    root = Path(__file__).resolve().parent
    checked = [
        "audio_synth.py",
        "m4b_assemble.py",
        "audiobook_verify.py",
        "audiobook_request.py",
        "hour_continue.py",
        "movie_assemble.py",
    ]
    offenders = []
    for name in checked:
        text = (root / name).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped in ('"ffmpeg",', '"ffprobe",', '["ffmpeg"', '["ffprobe"'):
                offenders.append(f"{name}:{line_number}: {stripped}")
    assert offenders == [], "bare media binary names remain: " + "; ".join(offenders)


# ##################################################################
# ffmpeg and ffprobe resolve the two distinct names
def test_each_accessor_resolves_its_own_binary() -> None:
    assert Path(ffmpeg()).name.startswith("ffmpeg")
    assert Path(ffprobe()).name.startswith("ffprobe")
    assert resolve("ffmpeg") == ffmpeg()
