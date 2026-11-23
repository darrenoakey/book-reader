"""Idempotent, asynchronous assistant audiobook requests.

``./run audiobook request "Title" [--author A] [--after "Earlier Title"]``
returns at once with a job record; a detached worker resolves nothing new and
produces, verifies and delivers in the background. Repeating a request returns
the same job (and revives a dead worker), never a second production or a
second Apple Books import.

A job is keyed by the WORK it narrates (normalized title + first author from
the EPUB's own metadata), stored atomically as ``jobs/<work_id>.json`` under
``[audiobook] delivery_root``. A job can adopt a live render, verify and
publish its result, copy it into private iCloud Drive, then wait for the
02:00–05:00 local window before importing into Apple Books on this Mac.
Publication, the local iCloud Files copy, and the verified Books library row
are distinct facts; none claims remote upload or iPhone visibility.

``--after`` names a prior requested book. The sequel waits for its verified
publication, then reuses recorded identities and byte-identical voice clips.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from src.audiobook_library import (
    author_key,
    load_index,
    resolve_book,
    title_key,
    tokens,
)
from src.books_delivery import deliver_to_books, find_delivered_asset, sha256_file

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "local" / "config.toml"
REQUEST_NAME = "request.json"
RECEIPT_NAME = "delivery_receipt.json"
QUIET_HOURS = range(2, 5)
WAIT_SECONDS = 60
TERMINAL = ("delivered", "published", "failed")
MILESTONES = ("extract", "characters", "voices_desc", "voices_clone", "scripts", "audio", "m4b")


# ##################################################################
# settings
# every location is explicit configuration; a missing or unmounted root fails closed
def settings(path: Path = CONFIG_PATH) -> dict[str, Path]:
    with path.open("rb") as stream:
        section = tomllib.load(stream).get("audiobook", {})
    resolved = {}
    for key in ("productions_root", "delivery_root", "library_search_root"):
        value = section.get(key)
        if not value:
            raise ValueError(f"[audiobook] {key} is not configured in {path}")
        root = Path(value).expanduser()
        if not root.is_dir():
            raise ValueError(f"configured {key} {root} does not exist or is not mounted")
        resolved[key] = root
    if len({root.resolve() for root in resolved.values()}) != len(resolved):
        raise ValueError("audiobook production, delivery and source roots must differ")
    if section.get("icloud_audiobooks_root"):
        resolved["icloud_audiobooks_root"] = Path(section["icloud_audiobooks_root"]).expanduser()
    return resolved


# ##################################################################
# atomic json
def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    partial = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now(UTC).isoformat()


# ##################################################################
# names
def project_slug(source: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", source.stem).strip("_")


def display_author(requested: str | None, metadata_authors: list[str]) -> str:
    """The request's spelling when it names the same person as the metadata
    ("Anne McCaffrey" for an EPUB tagged "McCaffrey Anne"), else the metadata."""
    first = metadata_authors[0] if metadata_authors else ""
    if requested and (not first or sorted(tokens(requested)) == sorted(tokens(first))):
        return requested.strip()
    return first or (requested or "Unknown Author").strip()


def publication_name(title: str, author: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "", f"{title} - {author}").strip()


# ##################################################################
# job store
def jobs_dir(config: dict[str, Path]) -> Path:
    return config["delivery_root"] / "jobs"


def job_path(config: dict[str, Path], job_id: str) -> Path:
    if not re.fullmatch(r"work-[0-9a-f]{16}", job_id):
        raise ValueError(f"not a job id: {job_id!r}")
    return jobs_dir(config) / f"{job_id}.json"


@contextmanager
def job_lock(config: dict[str, Path], job_id: str, *, worker: bool = False):
    path = job_path(config, job_id).with_suffix(".worker.lock" if worker else ".request.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock files are never unlinked: contenders must acquire the same inode.
    with path.open("ab") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if worker else 0))
        yield


def load_job(config: dict[str, Path], job_id: str) -> dict:
    path = job_path(config, job_id)
    if not path.is_file():
        raise FileNotFoundError(f"no audiobook job {job_id}")
    job = read_json(path)
    if job.get("job_id") != job_id:
        raise ValueError(f"job record {path.name} has a different identity")
    return job


def save_job(config: dict[str, Path], job: dict, **changes) -> dict:
    job = {**job, **changes, "updated": now()}
    write_json(job_path(config, job["job_id"]), job)
    return job


def all_jobs(config: dict[str, Path]) -> list[dict]:
    return [load_job(config, path.stem) for path in sorted(jobs_dir(config).glob("work-*.json"))]


def worker_alive(job: dict) -> bool:
    pid = job.get("worker_pid")
    if not pid or job.get("worker_host") != os.uname().nodename:
        return False
    return live_pid(pid, f"src.audiobook_request work {job['job_id']}")


def live_pid(pid: int | str, command: str | None = None) -> bool:
    try:
        pid = int(pid)
        if pid <= 1:
            return False
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return command is None
    if command is None:
        return True
    ps = subprocess.run(["ps", "-p", str(pid), "-o", "command="], check=False, capture_output=True, text=True, timeout=5)
    return ps.returncode == 0 and command in ps.stdout


# ##################################################################
# resolve a request to one work
def resolve_request(config: dict[str, Path], title: str, author: str | None) -> dict:
    index = load_index(config["library_search_root"], config["delivery_root"])
    found = resolve_book(title, author, index)
    if found["status"] == "not_found":
        # a stale index must not hide a newly added book
        found = resolve_book(title, author, load_index(config["library_search_root"], config["delivery_root"], refresh=True))
    return found


# ##################################################################
# request
# create (or return) the job for one book and make sure a worker owns it
def request(
    title: str,
    author: str | None = None,
    after: str | None = None,
    deliver: bool = True,
    start: bool = True,
    config_path: Path = CONFIG_PATH,
    series: str | None = None,
    series_required_characters: list[str] | None = None,
) -> dict:
    config = settings(config_path)
    existing = exact_job(config, title, author)
    found = None if existing else resolve_request(config, title, author)
    if found is not None and found["status"] != "resolved":
        return found
    work_id = existing["job_id"] if existing else found["work_id"]
    parent_id = None
    if after:
        parent = exact_job(config, after, author)
        if parent is None:
            parent_found = resolve_request(config, after, author)
            if parent_found["status"] != "resolved":
                return {"status": "series_parent_unresolved", "after": after, "resolution": parent_found}
            if not job_path(config, parent_found["work_id"]).is_file():
                return {"status": "series_parent_not_requested", "after": after, "job_id": parent_found["work_id"]}
            parent = load_job(config, parent_found["work_id"])
        parent_id = parent["job_id"]
        if parent_id == work_id:
            raise ValueError("an audiobook cannot be its own series parent")
    with job_lock(config, work_id):
        path = job_path(config, work_id)
        if path.is_file():
            job = load_job(config, work_id)
            if found is not None and job["source_sha256"] != found["source_sha256"]:
                raise ValueError(f"job {work_id} is bound to a different source edition")
            source = Path(job["source"])
            if source.is_file() and job["state"] not in ("delivered", "published"):
                identity = source_identity(source)
                if identity != job.get("source_stat"):
                    if sha256_file(source) != job["source_sha256"]:
                        raise ValueError(f"job {work_id} is bound to a different source edition")
                    job = save_job(config, job, source_stat=identity)
            if parent_id is not None and job.get("series_parent_job") != parent_id:
                raise ValueError(f"job {work_id} already has series parent {job.get('series_parent_job')}")
            if series is not None:
                if job.get("series") not in (None, series):
                    raise ValueError(f"job {work_id} already belongs to series {job.get('series')!r}")
                changes = {}
                if job.get("series") is None:
                    changes["series"] = series
                if series_required_characters and not job.get("series_required_characters"):
                    changes["series_required_characters"] = list(series_required_characters)
                if changes:
                    job = save_job(config, job, **changes)
        else:
            source = Path(found["source"])
            book_author = display_author(author, found["authors"])
            job = save_job(
                config,
                {
                    "job_id": work_id,
                    "title": found["title"],
                    "author": book_author,
                    "source": str(source),
                    "source_sha256": found["source_sha256"],
                    "source_stat": source_identity(source),
                    "series_parent_job": parent_id,
                    "series": series,
                    "series_required_characters": list(series_required_characters or []),
                    "deliver": deliver,
                    "project": str(config["productions_root"] / project_slug(source)),
                    "publication": str(config["delivery_root"] / publication_name(found["title"], book_author)),
                    "state": "queued",
                    "requested": now(),
                },
            )
        done = job["state"] == "delivered" or (job["state"] == "published" and not job.get("deliver", True))
        if start and not done and not worker_alive(job):
            job = spawn_worker(config, job, config_path)
    return {"status": "accepted", "job": job_status(job)}


def source_identity(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def exact_job(config: dict[str, Path], title: str, author: str | None) -> dict | None:
    matches = [
        job for job in all_jobs(config)
        if title_key(job["title"]) == title_key(title)
        and (author is None or author_key(job["author"]) == author_key(author))
    ]
    return matches[0] if len(matches) == 1 and author is not None else None


# ##################################################################
# spawn worker
# a detached process that outlives the requesting assistant turn
def spawn_worker(config: dict[str, Path], job: dict, config_path: Path = CONFIG_PATH) -> dict:
    project = Path(job["project"])
    project.mkdir(parents=True, exist_ok=True)
    log = project / "audiobook-worker.log"
    with log.open("ab") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.audiobook_request", "work", job["job_id"], "--config", str(config_path.resolve())],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state = "queued" if job["state"] == "failed" else job["state"]
    return save_job(config, job, worker_pid=process.pid, worker_host=os.uname().nodename, worker_log=str(log), state=state, error=None)


# ##################################################################
# bind request
# the project directory produces exactly one source file
def bind_request(project: Path, source: Path, source_sha256: str, series_parent: Path | None) -> dict:
    record = {"source": str(source), "source_sha256": source_sha256, "series_parent": str(series_parent) if series_parent else None}
    path = project / REQUEST_NAME
    if path.is_file():
        existing = read_json(path)
        for key in ("source_sha256", "series_parent"):
            if existing.get(key) != record[key]:
                raise ValueError(f"{project.name} is bound to a different {key} ({existing.get(key)}); refusing to mix productions")
        return existing
    project.mkdir(parents=True, exist_ok=True)
    write_json(path, {**record, "requested": now()})
    return record


# ##################################################################
# probe m4b
# real ffprobe evidence that the audiobook is complete and well-formed
def probe_m4b(m4b: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-show_chapters", str(m4b)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    cover = [s for s in streams if (s.get("disposition") or {}).get("attached_pic") == 1]
    duration = float(info.get("format", {}).get("duration") or 0)
    chapters = info.get("chapters", [])
    tags = info.get("format", {}).get("tags") or {}
    problems = []
    if not audio:
        problems.append("no audio stream")
    if duration < 60:
        problems.append(f"duration {duration:.0f}s")
    if not chapters:
        problems.append("no chapters")
    if not cover:
        problems.append("no embedded cover")
    if not tags.get("title"):
        problems.append("no title tag")
    if problems:
        raise ValueError(f"{m4b.name} failed verification: {', '.join(problems)}")
    return {
        "duration_seconds": round(duration, 1),
        "chapters": len(chapters),
        "title": tags.get("title"),
        "artist": tags.get("artist"),
        "cover": True,
    }


# ##################################################################
# publish
# copy the verified M4B and its cover into the delivery folder, sha-checked
def publish(m4b: Path, cover: Path, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    published = {}
    for source, name in ((m4b, f"{folder.name}.m4b"), (cover, "cover.jpg")):
        target = folder / name
        digest = sha256_file(source)
        if not (target.is_file() and sha256_file(target) == digest):
            if target.exists() or target.is_symlink():
                raise ValueError(f"refusing to replace different publication bytes at {target}")
            fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=folder)
            os.close(fd)
            partial = Path(temporary)
            try:
                shutil.copyfile(source, partial)
                if sha256_file(partial) != digest:
                    raise ValueError(f"published copy of {source.name} does not match its source bytes")
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        published[name] = {"path": str(target), "sha256": digest, "bytes": target.stat().st_size}
    return published


# ##################################################################
# books window

def books_window_open(moment: datetime | None = None) -> bool:
    moment = moment or datetime.now()
    return moment.hour in QUIET_HOURS


# ##################################################################
# existing receipt
# a prior receipt still backed by the same bytes in Books, else None
def existing_receipt(publication: Path, m4b: Path) -> dict | None:
    path = publication / RECEIPT_NAME
    if not path.is_file() or not m4b.is_file():
        return None
    receipt = read_json(path)
    if receipt.get("m4b_sha256") != sha256_file(m4b):
        return None
    asset = find_delivered_asset(m4b)
    if asset is None or asset["asset_id"] != receipt.get("books", {}).get("asset_id"):
        return None
    cloud = receipt.get("icloud_files", {})
    cloud_file = cloud.get("path")
    if not cloud_file or not Path(cloud_file).is_file() or sha256_file(Path(cloud_file)) != asset["sha256"]:
        return None
    return receipt


# ##################################################################
# work
# the detached worker: wait for series parent, produce, publish, deliver
def work(job_id: str, config_path: Path = CONFIG_PATH) -> dict:
    config = settings(config_path)
    try:
        with job_lock(config, job_id, worker=True):
            with job_lock(config, job_id):
                job = load_job(config, job_id)
                job = save_job(config, job, worker_pid=os.getpid(), worker_host=os.uname().nodename)
            try:
                return _work(config, job)
            except BaseException as error:
                save_job(config, load_job(config, job_id), state="failed", error=f"{type(error).__name__}: {error}")
                raise
    except BlockingIOError:
        if worker_alive(load_job(config, job_id)):
            return load_job(config, job_id)
        raise


def _work(config: dict[str, Path], job: dict) -> dict:
    project, publication = Path(job["project"]), Path(job["publication"])
    source = Path(job["source"])
    published_m4b = publication / f"{publication.name}.m4b"
    receipt = existing_receipt(publication, published_m4b)
    if receipt is not None:
        if receipt.get("job_id") != job["job_id"] or receipt.get("source_sha256") != job["source_sha256"]:
            raise ValueError(f"publication directory for {job['title']} contains a different work")
        return save_job(config, job, state="delivered", receipt=str(publication / RECEIPT_NAME), error=None)
    parent_project = None
    if job.get("series_parent_job"):
        parent_project = _await_series_parent(config, job)
    if sha256_file(source) != job["source_sha256"]:
        raise ValueError(f"{source} changed since it was requested")
    bind_request(project, source, job["source_sha256"], parent_project)
    if parent_project is not None:
        from src.series_voices import set_series_parent
        from src.tts_engine import engine_name

        if engine_name() != "breeze":
            raise ValueError("series voice reuse requires the breeze TTS engine")
        set_series_parent(project, parent_project)
    if pipeline_pid(project) is not None:
        job = save_job(config, job, state="adopting_existing_render")
        _await_project_unlocked(project)
    job = save_job(config, job, state="producing")
    from src.pipeline import run_audiobook_pipeline

    m4b = run_audiobook_pipeline(source, project, display_author=job["author"])
    from src.audiobook_cover import COVER_NAME
    from src.audiobook_verify import load_manifest, verify_audiobook
    from src.m4b_assemble import AUDIOBOOK_ASSEMBLY_NAME

    manifest = project / AUDIOBOOK_ASSEMBLY_NAME
    expected = load_manifest(manifest)
    if (expected.title, expected.author) != (job["title"], job["author"]):
        raise ValueError("assembly title or author differs from the requested work")
    report = verify_audiobook(m4b, manifest, require_byte_identical_cover=True)
    report.raise_for_status()
    probe = report.as_dict()
    write_json(project / "audiobook_verification.json", probe)
    published = publish(m4b, project / COVER_NAME, publication)
    job = save_job(config, job, state="published", probe=probe, published=published)
    if job.get("series"):
        # a published series parent becomes the voice contract its sequels inherit
        from src.series_voices import publish_series

        characters = json.loads((project / "characters.json").read_text(encoding="utf-8"))
        series_entry = publish_series(
            config["delivery_root"],
            job["series"],
            project,
            character_names={cid: info.get("name", cid) for cid, info in characters.items() if isinstance(info, dict)},
            required_characters=job.get("series_required_characters"),
        )
        job = save_job(config, job, series_registry=series_entry)
    if not job.get("deliver", True):
        return job
    from src.books_delivery import copy_to_icloud_files, icloud_delivery_evidence

    cloud_root = config.get("icloud_audiobooks_root")
    if cloud_root is None:
        raise ValueError("[audiobook] icloud_audiobooks_root is required for private delivery")
    cloud = icloud_delivery_evidence(copy_to_icloud_files(published_m4b, cloud_root))
    job = save_job(config, job, state="icloud_files_copied", icloud_files=cloud)
    asset = find_delivered_asset(published_m4b)
    if asset is None:
        job = save_job(config, job, state="waiting_for_books_window")
        lock = jobs_dir(config) / ".mac-books-import.lock"
        while asset is None:
            if books_window_open():
                with lock.open("ab") as guard:
                    fcntl.flock(guard, fcntl.LOCK_EX)
                    if books_window_open():
                        asset = deliver_to_books(published_m4b)
            if asset is None:
                time.sleep(WAIT_SECONDS)
    else:
        asset = {**asset, "imported_now": False}
    receipt = {
        "job_id": job["job_id"],
        "title": job["title"],
        "author": job["author"],
        "source": job["source"],
        "source_sha256": job["source_sha256"],
        "series_parent_job": job.get("series_parent_job"),
        "m4b": str(published_m4b),
        "m4b_sha256": asset["sha256"],
        "probe": probe,
        "published": published,
        "icloud_files": cloud,
        "books": {k: asset[k] for k in ("asset_id", "title", "stored_file", "bytes", "imported_now")},
        "verified": now(),
    }
    write_json(publication / RECEIPT_NAME, receipt)
    write_json(project / RECEIPT_NAME, receipt)
    return save_job(config, job, state="delivered", receipt=str(publication / RECEIPT_NAME), error=None)


def _await_project_unlocked(project: Path) -> None:
    # another live process (an earlier worker or a manual run) owns the project;
    # resume after it rather than fail or run two productions at once
    while pipeline_pid(project) is not None:
        time.sleep(WAIT_SECONDS)


def pipeline_pid(project: Path) -> int | None:
    try:
        pid = int((project / ".pipeline.lock").read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if live_pid(pid) else None


def _await_series_parent(config: dict[str, Path], job: dict) -> Path:
    while True:
        parent = load_job(config, job["series_parent_job"])
        if parent.get("published") and parent.get("probe", {}).get("ok"):
            return Path(parent["project"]).resolve()
        if parent["state"] == "failed" and not worker_alive(parent):
            raise ValueError(f"series parent {parent['title']} did not publish: {parent.get('error')}")
        if job["state"] != "waiting_for_series_parent":
            job = save_job(config, job, state="waiting_for_series_parent")
        time.sleep(WAIT_SECONDS)


# ##################################################################
# status
def status(job_id: str | None = None, config_path: Path = CONFIG_PATH) -> list[dict]:
    config = settings(config_path)
    jobs = [load_job(config, job_id)] if job_id else all_jobs(config)
    return [job_status(job) for job in jobs]


def job_status(job: dict) -> dict:
    from src.state import is_step_complete

    project = Path(job["project"])
    pid = pipeline_pid(project)
    current = next((step for step in MILESTONES if not is_step_complete(project, step)), None)
    progress = {"pipeline_pid": pid, "next_pipeline_stage": current}
    if current == "audio" and (project / "audio").is_dir():
        progress["wav_segments_on_disk"] = len(list((project / "audio").rglob("*.wav")))
    return {**job, "worker_alive": worker_alive(job), "progress": progress}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="audiobook")
    sub = parser.add_subparsers(dest="command", required=True)
    p_request = sub.add_parser("request", help="request an audiobook by title (returns at once)")
    p_request.add_argument("title")
    p_request.add_argument("--author")
    p_request.add_argument("--after", help="earlier book in the same series whose voices this one reuses")
    p_request.add_argument("--no-deliver", action="store_true", help="produce and publish only; no iCloud or Apple Books delivery")
    p_request.add_argument("--no-start", action="store_true", help="record the job without starting the worker")
    p_request.add_argument("--series", help="series this book belongs to; a published parent becomes its voice contract")
    p_request.add_argument(
        "--series-required",
        action="append",
        default=None,
        help="character display name the series contract requires in every volume (repeatable)",
    )
    p_status = sub.add_parser("status", help="show one job or all jobs")
    p_status.add_argument("job_id", nargs="?")
    p_work = sub.add_parser("work", help="(internal) run one job's detached worker")
    p_work.add_argument("job_id")
    p_work.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.command == "request":
        result = request(
            args.title,
            args.author,
            args.after,
            deliver=not args.no_deliver,
            start=not args.no_start,
            series=args.series,
            series_required_characters=args.series_required,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "accepted" else 2
    if args.command == "status":
        print(json.dumps(status(args.job_id), indent=2))
        return 0
    job = work(args.job_id, args.config)
    print(json.dumps(job, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "bind_request", "books_window_open", "display_author", "existing_receipt", "probe_m4b", "publication_name",
    "publish", "request", "settings", "status", "work",
]
