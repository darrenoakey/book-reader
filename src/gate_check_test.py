import subprocess
from pathlib import Path

from src.gate_check import MOVIE_TESTS, changed_paths, select_tests


# ##################################################################
# test impact selection
# Movie changes include title-card and entrypoint coverage; unrelated paths
# and unavailable history must retain the full pre-existing repository check.
def test_select_tests() -> None:
    assert select_tests(["src/movie_assemble.py", "README.md"]) == MOVIE_TESTS
    assert select_tests(["run", "src/pipeline.py", "src/step_runner.py"]) == MOVIE_TESTS
    assert select_tests(["src/movie_assemble.py", "src/audio_synth.py"]) is None
    assert select_tests(["requirements.txt"]) is None
    assert select_tests([]) is None
    assert select_tests(["README.md"]) is None


# ##################################################################
# test real git change detection
# Use an isolated real git repository to prove last-green comparison and the
# fail-closed behavior when no gate history exists.
def test_changed_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert changed_paths(tmp_path) == []
    source = tmp_path / "source.txt"
    source.write_text("first\n")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    commit = ["git", "-c", "user.name=Book Reader Tests", "-c", "user.email=tests@example.invalid",
              "-c", "commit.gpgsign=false", "commit", "-qm"]
    subprocess.run([*commit, "initial"], cwd=tmp_path, check=True)
    subprocess.run(["git", "update-ref", "refs/greenline/last-green", "HEAD"], cwd=tmp_path, check=True)
    assert changed_paths(tmp_path) == []
    source.write_text("second\n")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    subprocess.run([*commit, "changed"], cwd=tmp_path, check=True)
    assert changed_paths(tmp_path) == ["source.txt"]
