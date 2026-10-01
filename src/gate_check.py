# Change-impact gate for the movie assembly surface; unrelated changes retain
# the full repository check, including live model integrations.
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOVIE_PATHS = {
    "run",
    "src/movie_assemble.py",
    "src/movie_resolution.py",
    "src/movie_assemble_test.py",
    "src/movie_resolution_test.py",
    "src/pipeline.py",
    "src/step_runner.py",
    "src/title_page_test.py",
    "src/gate_check.py",
    "src/gate_check_test.py",
    "src/dep_install.py",
    "src/dep_install_test.py",
    "src/hour_runner.py",
    "src/hour_runner_test.py",
    "src/hour_continue.py",
    "src/hour_continue_test.py",
    "src/hourly_spans.py",
    "src/hourly_spans_test.py",
    "src/llm.py",
    "src/llm_test.py",
    "src/script_generate.py",
    "src/script_generate_test.py",
    "src/movie_images.py",
    "src/movie_images_test.py",
    "src/arbiter_tts.py",
    "src/arbiter_tts_test.py",
    "src/audio_synth.py",
    "src/audio_synth_test.py",
    "src/breeze_voices.py",
    "src/character_analysis.py",
    "src/character_analysis_test.py",
    "src/server.py",
    "src/title_page.py",
    "src/movie_storyboard.py",
    "src/movie_storyboard_test.py",
    "src/text_ingest.py",
    "src/text_ingest_test.py",
    "greenline.toml",
}
MOVIE_TESTS = (
    "src/movie_assemble_test.py",
    "src/movie_resolution_test.py",
    "src/title_page_test.py",
    "src/state_test.py",
    "src/gate_check_test.py",
    "src/dep_install_test.py",
    "src/hour_runner_test.py",
    "src/hourly_spans_test.py",
    "src/text_ingest_test.py",
    "src/movie_storyboard_test.py",
    "src/movie_images_test.py",
    "src/llm_test.py::test_load_local_toml_config",
    "src/script_generate_test.py",
)


# A path maps to the tests that execute its behavior. Importers are included
# only when their tests call the changed code. Unmapped movie-surface files
# still run the full movie set so a missing edge cannot drop coverage.
IMPACT: dict[str, tuple[str, ...]] = {
    "run": (
        "src/hour_runner_test.py::test_hour_verify_only_cli_uses_venv",
        "src/hour_continue_test.py",
        "src/movie_resolution_test.py",
        "src/dep_install_test.py",
        "src/gate_check_test.py",
    ),
    "greenline.toml": ("src/gate_check_test.py",),
    "src/movie_assemble.py": (
        "src/movie_assemble_test.py",
        "src/movie_resolution_test.py",
        "src/title_page_test.py",
    ),
    "src/movie_assemble_test.py": ("src/movie_assemble_test.py",),
    "src/movie_resolution.py": ("src/movie_resolution_test.py", "src/movie_assemble_test.py"),
    "src/movie_resolution_test.py": ("src/movie_resolution_test.py",),
    "src/pipeline.py": ("src/movie_resolution_test.py",),
    "src/step_runner.py": ("src/movie_resolution_test.py",),
    "src/title_page.py": ("src/title_page_test.py", "src/movie_assemble_test.py", "src/movie_resolution_test.py"),
    "src/title_page_test.py": ("src/title_page_test.py",),
    "src/gate_check.py": ("src/gate_check_test.py",),
    "src/gate_check_test.py": ("src/gate_check_test.py",),
    "src/dep_install.py": ("src/dep_install_test.py",),
    "src/dep_install_test.py": ("src/dep_install_test.py",),
    "src/hour_runner.py": ("src/hour_runner_test.py",),
    "src/hour_runner_test.py": ("src/hour_runner_test.py",),
    "src/hour_continue.py": ("src/hour_continue_test.py",),
    "src/hour_continue_test.py": ("src/hour_continue_test.py",),
    "src/hourly_spans.py": ("src/hourly_spans_test.py",),
    "src/hourly_spans_test.py": ("src/hourly_spans_test.py",),
    "src/llm.py": ("src/llm_test.py", "src/script_generate_test.py", "src/hourly_spans_test.py"),
    "src/llm_test.py": ("src/llm_test.py",),
    "src/script_generate.py": ("src/script_generate_test.py",),
    "src/script_generate_test.py": ("src/script_generate_test.py",),
    "src/movie_images.py": ("src/movie_images_test.py",),
    "src/movie_images_test.py": ("src/movie_images_test.py",),
    "src/arbiter_tts.py": ("src/arbiter_tts_test.py", "src/movie_images_test.py"),
    "src/arbiter_tts_test.py": ("src/arbiter_tts_test.py",),
    "src/audio_synth.py": ("src/audio_synth_test.py",),
    "src/audio_synth_test.py": ("src/audio_synth_test.py",),
    "src/breeze_voices.py": ("src/hour_runner_test.py",),
    "src/movie_storyboard.py": ("src/movie_storyboard_test.py",),
    "src/movie_storyboard_test.py": ("src/movie_storyboard_test.py",),
    "src/text_ingest.py": ("src/text_ingest_test.py",),
    "src/text_ingest_test.py": ("src/text_ingest_test.py",),
    "src/state_test.py": ("src/state_test.py",),
}


# ##################################################################
# select tests
# Union the tests that execute each changed path. Unknown code fails closed
# to the full repository check. A movie-surface file without a finer map keeps
# the full movie set rather than silently shrinking coverage.
def select_tests(paths: list[str]) -> tuple[str, ...] | None:
    code = [path for path in paths if not path.endswith(".md")]
    if not code or any(path not in MOVIE_PATHS for path in code):
        return None
    selected: list[str] = []
    for path in code:
        tests = IMPACT.get(path)
        if tests is None:
            return MOVIE_TESTS
        for test in tests:
            if test not in selected:
                selected.append(test)
    return tuple(selected)


# ##################################################################
# changed paths
# Read the gate candidate against last-green; absent history requires full checks.
def changed_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "refs/greenline/last-green", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return []
    return result.stdout.splitlines()


# ##################################################################
# main
# Lint the complete source tree and execute all impact-selected real integrations.
def _run_tests(tests: tuple[str, ...]) -> int:
    grouped: dict[str, list[str]] = {}
    for node in tests:
        grouped.setdefault(node.split("::", 1)[0], []).append(node)
    # One process per file. The files already use private temp dirs, so the
    # wall clock is the slowest file rather than the sum that blew the 180s budget.
    procs = [
        subprocess.Popen([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *nodes], cwd=ROOT)
        for nodes in grouped.values()
    ]
    rc = 0
    for proc in procs:
        code = proc.wait()
        if code:
            rc = code
    return rc


def main() -> int:
    tests = select_tests(changed_paths(ROOT))
    if tests is None:
        return subprocess.call(["dazpycheck"], cwd=ROOT)
    lint = subprocess.Popen([str(ROOT / "run"), "lint"], cwd=ROOT)
    test_rc = _run_tests(tests)
    lint_rc = lint.wait()
    return lint_rc or test_rc


# ##################################################################
# wait for json
# Restart returns before the inspect server binds. Retry until the deadline
# so a 2s startup is not a false deploy failure. Callers own the deadline:
# a standalone health probe must stay under five seconds.
def wait_for_json(url: str, deadline_s: float) -> tuple[int, object]:
    deadline = time.monotonic() + deadline_s
    last = "no response"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 1, last
        try:
            with urllib.request.urlopen(url, timeout=min(0.5, remaining)) as resp:
                return 0, json.loads(resp.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            last = str(exc)
        leftover = deadline - time.monotonic()
        if leftover <= 0:
            return 1, last
        time.sleep(min(0.25, leftover))


if __name__ == "__main__":
    raise SystemExit(main())
