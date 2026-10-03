import json
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Self

from src import server
from src.server import (
    PIPELINE_STEPS,
    STATIC_DIR,
    InspectHandler,
    build_static_hashes,
    project_detail,
    project_summary,
    resolve_static_tags,
)

ROOT = Path(__file__).resolve().parent.parent


# ##################################################################
# write jsonl
def write_jsonl(path: Path, records: list[dict], junk: bool = False) -> None:
    lines = [json.dumps(r) for r in records]
    if junk:
        lines.insert(1, "{not json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ##################################################################
# make project
# build a realistic project directory under root
def make_project(root: Path, name: str, full: bool = True) -> Path:
    project = root / name
    (project / "chapters").mkdir(parents=True)
    write_jsonl(
        project / "state.jsonl",
        [
            {"step": "extract", "status": "complete"},
            {"step": "characters", "detail": "complete"},
            {"step": "extract", "status": "complete"},
            {"step": "voices_desc", "status": "running"},
        ],
        junk=True,
    )
    write_jsonl(project / "timings.jsonl", [{"step": "extract", "seconds": 1.5}, {"step": "extract", "seconds": 2.5}])
    if full:
        (project / "refs").mkdir()
        (project / "refs" / "alice.png").write_bytes(b"png")
        (project / "voices").mkdir()
        (project / "voices" / "alice.wav").write_bytes(b"wav")
        (project / "characters.json").write_text(
            json.dumps({"alice": {"name": "Alice", "bio": "b" * 500}, "bob": {"description": "Bob desc"}}),
            encoding="utf-8",
        )
        (project / "scenes").mkdir()
        (project / "scenes" / "0001.png").write_bytes(b"img")
        (project / "storyboard.json").write_text(
            json.dumps(
                {
                    "style": "noir",
                    "scenes": [
                        {"index": 1, "start": 0, "end": 5, "characters": ["alice"], "prompt": "p", "text_excerpt": "t"},
                        {"index": 2, "start": 5, "end": 9, "characters": [], "prompt": "q", "text_excerpt": "u"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (project / "book.m4b").write_bytes(b"0123456789" * 100)
        (project / "movie").mkdir()
        (project / "movie" / "movie.mp4").write_bytes(b"mp4")
        (project / "audio").mkdir()
        for stem in ("001", "002", "001.announce", "003.tmp"):
            (project / "audio" / f"{stem}.wav").write_bytes(b"w")
    return project


# ##################################################################
# live server
# real inspect server on an OS-assigned port serving a temporary root
class LiveServer:
    def __init__(self, root: Path) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(InspectHandler, root=root))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> Self:
        build_static_hashes()
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def get(self, path: str, headers: dict | None = None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers), err.read()


# ##################################################################
# test static tags
# static markers are rewritten with the content hash of the real static file
def test_resolve_static_tags_uses_content_hash() -> None:
    build_static_hashes()
    css_hash = server._STATIC_HASHES["app.css"]
    assert len(css_hash) == 12
    out = resolve_static_tags('<link href="{{ static:app.css }}"><script src="{{static:missing.js}}">')
    assert f"/static/app.css?v={css_hash}" in out
    assert "/static/missing.js?v=dev" in out
    assert "{{" not in out


# ##################################################################
# test project summary
# summary reports completed steps (deduped, junk tolerated), next step and timings
def test_project_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = make_project(Path(tmp), "book-a")
        summary = project_summary(project)
        assert summary["name"] == "book-a"
        assert summary["steps_done"] == ["extract", "characters"]
        assert summary["steps_total"] == len(PIPELINE_STEPS)
        assert summary["next_step"] == "voices_desc"
        assert summary["has_audiobook"] and summary["has_movie"]
        assert summary["scenes"] == 2
        assert summary["timings"] == {"extract": 2.5}


# ##################################################################
# test empty project summary
# a project with no artifacts yields empty defaults
def test_project_summary_empty_project() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "bare"
        project.mkdir()
        summary = project_summary(project)
        assert summary["steps_done"] == []
        assert summary["next_step"] == PIPELINE_STEPS[0]
        assert not summary["has_audiobook"] and not summary["has_movie"]
        assert summary["scenes"] == 0 and summary["timings"] == {}


# ##################################################################
# test corrupt json tolerated
# a corrupt storyboard is treated as absent rather than crashing the listing
def test_corrupt_json_is_tolerated() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = make_project(Path(tmp), "corrupt", full=False)
        (project / "storyboard.json").write_text("{broken", encoding="utf-8")
        (project / "characters.json").write_text("[1,2", encoding="utf-8")
        detail = project_detail(project)
        assert detail["scenes"] == [] and detail["characters"] == [] and detail["style"] == ""


# ##################################################################
# test project detail
# detail links existing refs/voices/scenes/media and lists only plain chapters
def test_project_detail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = make_project(Path(tmp), "book-b")
        detail = project_detail(project)
        alice, bob = detail["characters"]
        assert alice["id"] == "alice" and alice["name"] == "Alice" and len(alice["bio"]) == 400
        assert alice["ref_image"] == "/media/book-b/refs/alice.png"
        assert alice["voice_clip"] == "/media/book-b/voices/alice.wav"
        assert bob["name"] == "bob" and bob["bio"] == "Bob desc"
        assert bob["ref_image"] is None and bob["voice_clip"] is None
        assert detail["style"] == "noir"
        assert [s["image"] for s in detail["scenes"]] == ["/media/book-b/scenes/0001.png", None]
        assert detail["audiobook_url"] == "/media/book-b/book.m4b"
        assert detail["movie_url"] == "/media/book-b/movie/movie.mp4"
        assert detail["chapters"] == ["001", "002"]


# ##################################################################
# test detail without media
def test_project_detail_without_media() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        detail = project_detail(make_project(Path(tmp), "plain", full=False))
        assert detail["audiobook_url"] is None and detail["movie_url"] is None
        assert detail["chapters"] == []


# ##################################################################
# test http index and static
# the index page and static assets are served with correct caching headers
def test_http_index_and_static() -> None:
    with tempfile.TemporaryDirectory() as tmp, LiveServer(Path(tmp)) as srv:
        status, headers, body = srv.get("/")
        page = body.decode("utf-8")
        assert status == 200 and headers["Content-Type"] == "text/html"
        assert headers["Cache-Control"] == "no-store"
        assert "{{" not in page and "/static/app.css?v=" in page
        status, headers, body = srv.get("/static/app.css")
        assert status == 200 and headers["Content-Type"] == "text/css"
        assert "immutable" in headers["Cache-Control"]
        assert body == (STATIC_DIR / "app.css").read_bytes()
        assert srv.get("/static/nope.css")[0] == 404
        assert srv.get("/static/.hidden")[0] == 404
        assert srv.get("/static/a/b.css")[0] == 404
        status, _, body = srv.get("/static/..%2Frun")
        assert status == 404 and json.loads(body)["error"] == "bad static path"


# ##################################################################
# test http api
# project list contains only chapter-bearing projects, detail 404s for unknown
def test_http_api() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root, "listed", full=False)
        (root / "not-a-project").mkdir()
        (root / "stray.txt").write_text("x")
        with LiveServer(root) as srv:
            status, headers, body = srv.get("/api/projects")
            assert status == 200 and headers["Content-Type"] == "application/json"
            assert [p["name"] for p in json.loads(body)] == ["listed"]
            status, _, body = srv.get("/api/project/listed")
            assert status == 200 and json.loads(body)["name"] == "listed"
            status, _, body = srv.get("/api/project/ghost")
            assert status == 404 and json.loads(body)["error"] == "project not found"
            assert srv.get("/nothing/here")[0] == 404


# ##################################################################
# test http projects newest first
def test_http_projects_sorted_newest_first() -> None:
    import os

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old = make_project(root, "old", full=False)
        new = make_project(root, "new", full=False)
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        with LiveServer(root) as srv:
            names = [p["name"] for p in json.loads(srv.get("/api/projects")[2])]
            assert names == ["new", "old"]


# ##################################################################
# test http media
# media is served whole, range-aware, with correct types and traversal blocked
def test_http_media() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root, "media")
        (root / "secret.txt").write_text("top secret")
        data = b"0123456789" * 100
        with LiveServer(root) as srv:
            status, headers, body = srv.get("/media/media/book.m4b")
            assert status == 200 and body == data
            assert headers["Content-Type"] == "audio/mp4" and headers["Accept-Ranges"] == "bytes"
            assert srv.get("/media/media/refs/alice.png")[1]["Content-Type"] == "image/png"
            status, headers, body = srv.get("/media/media/book.m4b", {"Range": "bytes=10-19"})
            assert status == 206 and body == data[10:20]
            assert headers["Content-Range"] == "bytes 10-19/1000"
            status, headers, body = srv.get("/media/media/book.m4b", {"Range": "bytes=990-"})
            assert status == 206 and body == data[990:] and headers["Content-Range"] == "bytes 990-999/1000"
            status, headers, body = srv.get("/media/media/book.m4b", {"Range": "bytes=900-5000"})
            assert status == 206 and body == data[900:]
            status, _, body = srv.get("/media/media/book.m4b", {"Range": "garbage"})
            assert status == 200 and body == data
            assert srv.get("/media/media/missing.wav")[0] == 404
            assert srv.get("/media/media/movie")[0] == 404
            assert srv.get("/media/media/..%2Fsecret.txt")[0] == 404
            assert srv.get("/media/ghost/book.m4b")[0] == 404


# ##################################################################
# test cli serves on os port
# the real entry point binds an OS-assigned port, announces it and serves the UI
def test_cli_serves_on_os_assigned_port() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "src.server", "--port", "0"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert line.startswith("book-reader inspect UI on http://127.0.0.1:"), line
        port = int(line.rsplit(":", 1)[1])
        assert port > 0
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as resp:
            assert resp.status == 200
            assert "Book Reader" in resp.read().decode("utf-8")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        proc.stdout.close()
