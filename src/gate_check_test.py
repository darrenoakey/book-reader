import ast
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.gate_check import (
    IMPACT,
    MOVIE_PATHS,
    RECOVERY_TESTS,
    changed_paths,
    select_tests,
    wait_for_json,
)

ROOT = Path(__file__).resolve().parent.parent


# ##################################################################
# test impact selection
# Movie changes select the tests that execute the changed modules. Unrelated
# paths and unavailable history must retain the full pre-existing repository check.
def test_select_tests() -> None:
    assert select_tests(["src/movie_assemble.py", "README.md"]) == (
        "src/movie_assemble_test.py",
        "src/movie_resolution_test.py",
        "src/title_page_test.py",
    )
    entry = select_tests(["run", "src/pipeline.py", "src/step_runner.py"])
    assert entry == (
        "src/hour_runner_test.py::test_hour_verify_only_cli_uses_venv",
        "src/hour_continue_test.py",
        "src/movie_resolution_test.py",
        "src/dep_install_test.py",
        "src/gate_check_test.py",
        "src/pipeline_test.py",
        "src/step_runner_test.py",
    )
    assert "src/movie_images_test.py" not in entry
    schema = select_tests(["src/hourly_spans.py", "src/hourly_spans_test.py", "src/llm.py"])
    assert schema == (
        "src/hourly_spans_test.py",
        "src/script_generate_test.py",
        "src/llm_test.py",
        "src/scriptor_attribution_test.py",
    )
    assert "src/movie_images_test.py" not in schema
    assert "src/movie_assemble_test.py" not in schema
    assert select_tests(["src/server.py"]) == ("src/server_test.py",)
    assert select_tests(["run", "src/dep_install.py", "src/dep_install_test.py"]) == (
        "src/hour_runner_test.py::test_hour_verify_only_cli_uses_venv",
        "src/hour_continue_test.py",
        "src/movie_resolution_test.py",
        "src/dep_install_test.py",
        "src/gate_check_test.py",
    )
    assert select_tests(["src/hour_continue.py", "src/hour_continue_test.py"]) == ("src/hour_continue_test.py",)
    assert select_tests(["src/audio_synth.py", "src/audio_synth_test.py"]) == (
        "src/audio_synth_test.py",
        "src/data_recovery_test.py",
    )
    assert select_tests(["src/arbiter_tts.py", "src/arbiter_tts_test.py"]) == (
        "src/arbiter_tts_test.py",
        "src/movie_images_test.py",
    )
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
    commit = [
        "git",
        "-c",
        "user.name=Book Reader Tests",
        "-c",
        "user.email=tests@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
    ]
    subprocess.run([*commit, "initial"], cwd=tmp_path, check=True)
    subprocess.run(["git", "update-ref", "refs/greenline/last-green", "HEAD"], cwd=tmp_path, check=True)
    assert changed_paths(tmp_path) == []
    source.write_text("second\n")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    subprocess.run([*commit, "changed"], cwd=tmp_path, check=True)
    assert changed_paths(tmp_path) == ["source.txt"]


# ##################################################################
# test probe retries until the server binds
# Deploy used to health-check once and fail while the inspect process was still starting.
def test_wait_for_json_retries_until_listening() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b"[1]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]

    def serve_late() -> None:
        time.sleep(0.8)
        server.serve_forever()

    thread = threading.Thread(target=serve_late, daemon=True)
    thread.start()
    try:
        rc, payload = wait_for_json(f"http://127.0.0.1:{port}/api/projects", 4.0)
        assert rc == 0
        assert payload == [1]
    finally:
        server.shutdown()
        server.server_close()


# ##################################################################
# test probe fails a closed port
# A down service must still fail, and a short deadline must not hang.
def test_wait_for_json_fails_closed_port() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    started = time.monotonic()
    rc, _message = wait_for_json(f"http://127.0.0.1:{port}/api/projects", 0.6)
    assert rc == 1
    assert time.monotonic() - started < 2


# ##################################################################
# import graph
# Parse the real source files so the selector is checked against actual
# imports rather than a second hand-maintained list.
def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("src."):
                found.add(module.split(".")[1])
            elif module == "src" or (node.level == 1 and not module):
                found.update(alias.name for alias in node.names)
            elif node.level == 1:
                found.add(module.split(".")[0])
        elif isinstance(node, ast.Import):
            found.update(a.name.split(".")[1] for a in node.names if a.name.startswith("src."))
    return found


def _consumers(target: str) -> set[str]:
    modules = {path.stem: _imports(path) for path in (ROOT / "src").glob("*.py")}
    reached: set[str] = set()
    pending = [target]
    while pending:
        current = pending.pop()
        for name, deps in modules.items():
            if current in deps and name not in reached:
                reached.add(name)
                pending.append(name)
    return reached


# ##################################################################
# test recovery consumers map to real suites
# Every module that transitively imports data_recovery, and every test that does,
# must be executed when data_recovery changes, using the real import graph.
def test_data_recovery_selects_all_transitive_consumers() -> None:
    suites = {f"src/{name}.py" for name in _consumers("data_recovery") if name.endswith("_test")}
    selected = select_tests(["src/data_recovery.py"])
    assert selected is not None
    assert set(selected) == set(RECOVERY_TESTS) == suites
    assert select_tests(["src/data_recovery_test.py"]) == ("src/data_recovery_test.py",)


# ##################################################################
# test consumers select their own suite and the recovery contract suite
# A consumer change must run its own test and, when the recovery suite imports
# the consumer, that suite too. Neither may fall back to the repository scanner.
def test_recovery_consumers_select_own_and_contract_suites() -> None:
    contract_imports = _imports(ROOT / "src" / "data_recovery_test.py")
    for name in sorted(_consumers("data_recovery")):
        if name.endswith("_test"):
            continue
        selected = select_tests([f"src/{name}.py"])
        assert selected is not None, name
        assert f"src/{name}_test.py" in selected, name
        if name in contract_imports:
            assert "src/data_recovery_test.py" in selected, name
        for test in selected:
            assert (ROOT / test.split("::")[0]).exists(), test


# ##################################################################
# test selector tables are internally consistent
# Every mapped path and test exists on disk and is inside the gated surface, so
# an entry cannot silently select nothing or fall outside the allowed set.
def test_impact_tables_reference_real_files() -> None:
    for path, tests in IMPACT.items():
        assert path in MOVIE_PATHS, path
        assert (ROOT / path).exists(), path
        for test in tests:
            assert (ROOT / test.split("::")[0]).exists(), test


# ##################################################################
# test branch test files and unknown paths
# Newly added co-located tests select themselves; unknown code stays fail-closed.
def test_branch_tests_select_themselves_and_unknown_fails_closed() -> None:
    for path in (
        "jeff_book_test.py",
        "src/conftest_test.py",
        "src/server_test.py",
        "src/kokoro_voices_test.py",
        "src/tts_engine_test.py",
        "src/voice_acoustic_test.py",
        "src/voice_clone_test.py",
        "src/breeze_voices_test.py",
        "src/demo_outputs_test.py",
        "src/pipeline_test.py",
        "src/step_runner_test.py",
    ):
        assert select_tests([path]) == (path,), path
    assert select_tests(["jeff_book.py"]) == ("jeff_book_test.py",)
    assert select_tests(["src/conftest.py"]) is None
    assert select_tests(["src/data_recovery.py", "src/not_a_real_module.py"]) is None


# ##################################################################
# test future recovery consumers cannot be missed
# Any source file that references data_recovery at all (import in any form or
# a dynamic/string reference) must be a graph consumer whose own suite is selected
# by a standalone data_recovery change. A module that mentions it without being
# in the graph, or a consumer whose suite is unmapped, fails here so the map
# must be extended with the new edge.
def test_every_data_recovery_reference_is_selected() -> None:
    selected = select_tests(["src/data_recovery.py"])
    assert selected is not None
    consumers = _consumers("data_recovery")
    for path in sorted((ROOT / "src").glob("*.py")):
        if path.stem in {"data_recovery", "gate_check", "gate_check_test"}:
            continue
        if "data_recovery" in path.read_text():
            assert path.stem in consumers, f"{path.name} references data_recovery outside the import graph"
    for name in sorted(consumers):
        suite = name if name.endswith("_test") else f"{name}_test"
        if (ROOT / "src" / f"{suite}.py").exists():
            assert f"src/{suite}.py" in selected, f"data_recovery change misses {suite}"
    # hourly_spans imports only llm and scriptor_attribution, so it is not a recovery consumer; its suite
    # runs through the script_generate edge instead of the recovery set.
    assert "hourly_spans" not in consumers
    assert "src/hourly_spans_test.py" in select_tests(["src/hourly_spans.py"])
    assert "src/script_generate_test.py" in selected
