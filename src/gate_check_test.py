import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.gate_check import MOVIE_TESTS, changed_paths, select_tests, wait_for_json


# ##################################################################
# test impact selection
# Movie changes include title-card and entrypoint coverage; unrelated paths
# and unavailable history must retain the full pre-existing repository check.
def test_select_tests() -> None:
    assert select_tests(["src/movie_assemble.py", "README.md"]) == MOVIE_TESTS
    assert select_tests(["run", "src/pipeline.py", "src/step_runner.py"]) == MOVIE_TESTS
    assert select_tests(["run", "src/dep_install.py", "src/dep_install_test.py"]) == MOVIE_TESTS
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
