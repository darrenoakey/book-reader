# Change-impact gate for the movie assembly surface; unrelated changes retain
# the full repository check, including live model integrations.
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOVIE_PATHS = {
    "run", "src/movie_assemble.py", "src/movie_resolution.py", "src/movie_assemble_test.py",
    "src/movie_resolution_test.py", "src/pipeline.py", "src/step_runner.py",
    "src/title_page_test.py", "src/gate_check.py", "src/gate_check_test.py",
    "greenline.toml",
}
MOVIE_TESTS = (
    "src/movie_assemble_test.py", "src/movie_resolution_test.py",
    "src/title_page_test.py", "src/state_test.py", "src/gate_check_test.py",
)


# ##################################################################
# select tests
# Select the complete movie/entrypoint/title/state surface for assembly changes;
# unknown code paths fail toward the full existing check, not reduced coverage.
def select_tests(paths: list[str]) -> tuple[str, ...] | None:
    code = [path for path in paths if not path.endswith(".md")]
    if not code or any(path not in MOVIE_PATHS for path in code):
        return None
    return MOVIE_TESTS


# ##################################################################
# changed paths
# Read the gate candidate against last-green; absent history requires full checks.
def changed_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "refs/greenline/last-green", "HEAD"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return []
    return result.stdout.splitlines()


# ##################################################################
# main
# Lint the complete source tree and execute all impact-selected real integrations.
def main() -> int:
    tests = select_tests(changed_paths(ROOT))
    if tests is None:
        return subprocess.call(["dazpycheck"], cwd=ROOT)
    lint = subprocess.call([str(ROOT / "run"), "lint"], cwd=ROOT)
    if lint:
        return lint
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
