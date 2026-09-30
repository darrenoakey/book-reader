import subprocess
import sys
import time
from pathlib import Path

from src.dep_install import (
    dependencies_current,
    dependency_fingerprint,
    ensure_installed,
    find_uv,
    install_requirements,
    write_stamp,
)


# ##################################################################
# test fingerprint tracks local dependency
# A file:// edit must invalidate the install stamp; an unchanged tree must not.
def test_fingerprint_tracks_local_dependency(tmp_path: Path) -> None:
    dep = tmp_path / "dep"
    dep.mkdir()
    module = dep / "mod.py"
    module.write_text("x = 1\n", encoding="utf-8")
    root = tmp_path / "proj"
    root.mkdir()
    (root / "requirements.txt").write_text(f"demo @ file://{dep}\n", encoding="utf-8")
    first = dependency_fingerprint(root)
    assert first == dependency_fingerprint(root)
    module.write_text("x = 2\n", encoding="utf-8")
    assert dependency_fingerprint(root) != first


# ##################################################################
# test stamp skip
# The gate skips install only for the fingerprint written into this venv.
def test_stamp_skip(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    assert not dependencies_current(venv, "abc")
    write_stamp(venv, "abc")
    assert dependencies_current(venv, "abc")
    assert not dependencies_current(venv, "def")


# ##################################################################
# test offline install from cache
# The release host must install a cached requirement without contacting PyPI.
def test_offline_install_from_cache(tmp_path: Path) -> None:
    assert find_uv() is not None, "uv is required so a cold gate does not re-download the world"
    subprocess.run([sys.executable, "-m", "venv", str(tmp_path / ".venv")], check=True)
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    python = tmp_path / ".venv" / "bin" / "python"
    started = time.monotonic()
    assert install_requirements(tmp_path, python) == 0
    assert time.monotonic() - started < 45
    assert subprocess.run([str(python), "-c", "import pytest"], check=False).returncode == 0
    stamp_started = time.monotonic()
    assert ensure_installed(tmp_path, tmp_path / ".venv", python) == 0
    assert time.monotonic() - stamp_started < 45
    skip_started = time.monotonic()
    assert ensure_installed(tmp_path, tmp_path / ".venv", python) == 0
    assert time.monotonic() - skip_started < 1
