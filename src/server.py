"""Inspect UI for book-reader projects.

A small stdlib HTTP server (per the website skill: hashed static files,
auto-managed port, no framework) that lists every project in output/ and
shows its pipeline status, characters and voices, scene storyboard, the M4B
audiobook, and the final movie — all playable/browsable in the browser.

Endpoints:
  GET /                        single-page UI
  GET /static/<file>           hashed, immutable static assets
  GET /api/projects            all projects with step status + artifacts
  GET /api/project/<name>      full detail for one project
  GET /media/<name>/<relpath>  project artifacts (Range-aware for av)
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
STATIC_DIR = PROJECT_ROOT / "static"
INSPECT_SERVICE = "book-reader-inspect"
INSPECT_PORT = 8769

PIPELINE_STEPS = [
    "extract", "characters", "voices_desc", "voices_clone",
    "scripts", "audio", "m4b", "storyboard", "refimages", "sceneimages", "movie",
]

CONTENT_TYPES = {
    ".css": "text/css",
    ".js": "application/javascript",
    ".html": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".wav": "audio/wav",
    ".m4b": "audio/mp4",
    ".mp4": "video/mp4",
}

_STATIC_HASHES: dict[str, str] = {}
_STATIC_TAG = re.compile(r"\{\{\s*static:([^}]+)\s*\}\}")


# ##################################################################
# build static hashes
# content-hash every static file once so browsers can cache them forever
def build_static_hashes() -> None:
    _STATIC_HASHES.clear()
    if not STATIC_DIR.is_dir():
        return
    for path in STATIC_DIR.iterdir():
        if path.is_file():
            _STATIC_HASHES[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# ##################################################################
# resolve static tags
# rewrite {{ static:file }} markers to cache-busted /static urls
def resolve_static_tags(page: str) -> str:
    return _STATIC_TAG.sub(lambda m: f"/static/{m.group(1).strip()}?v={_STATIC_HASHES.get(m.group(1).strip(), 'dev')}", page)


# ##################################################################
# load json
def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ##################################################################
# completed steps
# steps marked complete in state.jsonl (append-only progress log)
def _completed_steps(project: Path) -> list[str]:
    state = project / "state.jsonl"
    done: list[str] = []
    if state.exists():
        for line in state.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            if step and (rec.get("status") == "complete" or rec.get("detail") == "complete") and step not in done:
                done.append(step)
    return done


# ##################################################################
# latest timings
# most recent duration per step from timings.jsonl
def _timings(project: Path) -> dict[str, float]:
    timings: dict[str, float] = {}
    path = project / "timings.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("step"):
                timings[rec["step"]] = rec.get("seconds", 0.0)
    return timings


# ##################################################################
# project summary
def project_summary(project: Path) -> dict:
    done = _completed_steps(project)
    storyboard = _load_json(project / "storyboard.json") or {}
    movie = project / "movie" / "movie.mp4"
    m4bs = list(project.glob("*.m4b"))
    return {
        "name": project.name,
        "steps_done": done,
        "steps_total": len(PIPELINE_STEPS),
        "next_step": next((s for s in PIPELINE_STEPS if s not in done), None),
        "has_audiobook": bool(m4bs),
        "has_movie": movie.exists(),
        "scenes": len(storyboard.get("scenes", [])),
        "timings": _timings(project),
    }


# ##################################################################
# project detail
def project_detail(project: Path) -> dict:
    detail = project_summary(project)
    characters = _load_json(project / "characters.json") or {}
    detail["characters"] = [
        {
            "id": cid,
            "name": info.get("name", cid),
            "bio": (info.get("bio") or info.get("description") or "")[:400],
            "ref_image": f"/media/{project.name}/refs/{cid}.png" if (project / "refs" / f"{cid}.png").exists() else None,
            "voice_clip": f"/media/{project.name}/voices/{cid}.wav" if (project / "voices" / f"{cid}.wav").exists() else None,
        }
        for cid, info in sorted(characters.items())
    ]
    storyboard = _load_json(project / "storyboard.json") or {}
    detail["style"] = storyboard.get("style", "")
    detail["scenes"] = [
        {
            **{k: s.get(k) for k in ("index", "start", "end", "characters", "prompt", "text_excerpt")},
            "image": f"/media/{project.name}/scenes/{s['index']:04d}.png"
            if (project / "scenes" / f"{s['index']:04d}.png").exists()
            else None,
        }
        for s in storyboard.get("scenes", [])
    ]
    m4bs = list(project.glob("*.m4b"))
    detail["audiobook_url"] = f"/media/{project.name}/{m4bs[0].name}" if m4bs else None
    detail["movie_url"] = f"/media/{project.name}/movie/movie.mp4" if (project / "movie" / "movie.mp4").exists() else None
    detail["chapters"] = [
        p.stem for p in sorted((project / "audio").glob("*.wav")) if p.stem.endswith(".announce") is False and "." not in p.stem
    ] if (project / "audio").exists() else []
    return detail


# ##################################################################
# handler
class InspectHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, root: Path = OUTPUT_ROOT, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        pass

    # --------------------------------------------------------------
    def _send(self, body: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(json.dumps(obj).encode("utf-8"), "application/json")

    def _404(self, msg: str = "not found") -> None:
        body = json.dumps({"error": msg}).encode("utf-8")
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --------------------------------------------------------------
    # serve one artifact file with HTTP Range support (video/audio seeking)
    def _serve_media(self, name: str, relpath: str) -> None:
        project = (self.root / name).resolve()
        target = (project / relpath).resolve()
        if not str(target).startswith(str(project)) or not target.is_file():
            self._404("media not found")
            return
        size = target.stat().st_size
        ctype = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if m:
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with target.open("rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(remaining, 1 << 20))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with target.open("rb") as f:
            while chunk := f.read(1 << 20):
                self.wfile.write(chunk)

    # --------------------------------------------------------------
    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
            self._send(resolve_static_tags(page).encode("utf-8"), "text/html")
            return
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            if "/" in name or name.startswith("."):
                self._404("bad static path")
                return
            target = STATIC_DIR / name
            if not target.is_file():
                self._404("static not found")
                return
            ctype = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
            self._send(target.read_bytes(), ctype, cache="public, max-age=31536000, immutable")
            return
        if path == "/api/projects":
            projects = sorted(
                (p for p in self.root.iterdir() if p.is_dir() and (p / "chapters").exists()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            self._json([project_summary(p) for p in projects])
            return
        m = re.match(r"^/api/project/(.+)$", path)
        if m:
            project = self.root / m.group(1)
            if not project.is_dir():
                self._404("project not found")
                return
            self._json(project_detail(project))
            return
        m = re.match(r"^/media/([^/]+)/(.+)$", path)
        if m:
            self._serve_media(m.group(1), m.group(2))
            return
        self._404()


# ##################################################################
# serve
def serve(host: str = "127.0.0.1", port: int = INSPECT_PORT, root: Path = OUTPUT_ROOT) -> ThreadingHTTPServer:
    build_static_hashes()
    handler = partial(InspectHandler, root=root)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"book-reader inspect UI on http://{host}:{httpd.server_address[1]}")
    httpd.serve_forever()
    return httpd


# ##################################################################
# main
def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="book-reader-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=INSPECT_PORT)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
