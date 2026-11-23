"""The request layer against real files: EPUB packages, a real ffmpeg-built M4B,
an isolated config/delivery root, and the real (read-only) Apple Books library."""

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from src import audiobook_request as req
from src.audiobook_library_test import make_epub
from src.books_delivery import audiobook_assets, find_delivered_asset, sha256_file


def make_config(root: Path) -> Path:
    for name in ("productions", "delivery", "library"):
        (root / name).mkdir(parents=True, exist_ok=True)
    config = root / "config.toml"
    config.write_text(
        "[audiobook]\n"
        f'productions_root = "{root / "productions"}"\n'
        f'delivery_root = "{root / "delivery"}"\n'
        f'library_search_root = "{root / "library"}"\n',
        encoding="utf-8",
    )
    return config


def make_m4b(dest: Path, seconds: int = 75, cover: bool = True) -> Path:
    art = dest.with_suffix(".jpg")
    Image.new("RGB", (400, 400), (40, 70, 120)).save(art)
    chapters = dest.with_suffix(".meta")
    chapters.write_text(
        ";FFMETADATA1\ntitle=Test Book\nartist=Test Author\n"
        f"[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND={seconds * 500}\ntitle=One\n"
        f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={seconds * 500}\nEND={seconds * 1000}\ntitle=Two\n",
        encoding="utf-8",
    )
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}", "-i", str(chapters)]
    if cover:
        cmd += ["-i", str(art), "-map", "0:a", "-map", "2:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]
    else:
        cmd += ["-map", "0:a"]
    cmd += ["-map_metadata", "1", "-c:a", "aac", "-b:a", "32k", str(dest)]
    subprocess.run(cmd, check=True, timeout=120)
    return dest


def test_settings_require_every_configured_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = make_config(Path(tmp))
        assert set(req.settings(config)) == {"productions_root", "delivery_root", "library_search_root"}
        (Path(tmp) / "library").rmdir()
        with pytest.raises(ValueError, match="library_search_root"):
            req.settings(config)
        config.write_text('[audiobook]\nproductions_root = "/tmp"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="delivery_root is not configured"):
            req.settings(config)


def test_request_is_idempotent_and_keyed_by_work() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = make_config(root)
        make_epub(root / "library" / "Anne McCaffrey - Dragonsong.epub", "Dragonsong", "McCaffrey Anne")
        make_epub(root / "library" / "Anne McCaffrey - Dragonsinger.epub", "Dragonsinger", "McCaffrey Anne")
        first = req.request("Dragonsong", "Anne McCaffrey", start=False, config_path=config)
        assert first["status"] == "accepted"
        job = first["job"]
        assert job["state"] == "queued"
        assert job["author"] == "Anne McCaffrey"  # the request's spelling of the metadata's "McCaffrey Anne"
        assert job["publication"].endswith("Dragonsong - Anne McCaffrey")
        again = req.request("dragonsong", None, start=False, config_path=config)
        assert again["job"]["job_id"] == job["job_id"] and again["job"]["requested"] == job["requested"]
        assert len(req.status(config_path=config)) == 1

        sequel = req.request("Dragonsinger", "Anne McCaffrey", after="Dragonsong", start=False, config_path=config)
        assert sequel["job"]["series_parent_job"] == job["job_id"]
        orphan = req.request("Dragonsinger", None, after="Nonexistent Book", start=False, config_path=config)
        assert orphan["status"] == "series_parent_unresolved"

        # the same work from different bytes is never silently mixed into the job
        make_epub(root / "library" / "Anne McCaffrey - Dragonsong.epub", "Dragonsong", "McCaffrey Anne", {"OEBPS/x.txt": "new"})
        with pytest.raises(ValueError, match="different source"):
            req.request("Dragonsong", "Anne McCaffrey", start=False, config_path=config)
        assert req.request("Unknown Title", None, start=False, config_path=config)["status"] == "not_found"


def test_display_author_and_publication_name() -> None:
    assert req.display_author("Anne McCaffrey", ["McCaffrey Anne"]) == "Anne McCaffrey"
    assert req.display_author("Somebody Else", ["McCaffrey Anne"]) == "McCaffrey Anne"
    assert req.display_author(None, ["Jane Roe"]) == "Jane Roe"
    assert req.publication_name("What: If?", "A/B") == "What If - AB"


def test_bind_request_fails_closed_on_a_different_source_or_parent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "project"
        source = Path(tmp) / "book.epub"
        req.bind_request(project, source, "a" * 64, None)
        assert req.bind_request(project, source, "a" * 64, None)["source_sha256"] == "a" * 64
        with pytest.raises(ValueError, match="source_sha256"):
            req.bind_request(project, source, "b" * 64, None)
        with pytest.raises(ValueError, match="series_parent"):
            req.bind_request(project, source, "a" * 64, Path(tmp) / "parent")


def test_probe_and_publish_a_real_m4b() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        m4b = make_m4b(Path(tmp) / "book.m4b")
        probe = req.probe_m4b(m4b)
        assert probe["chapters"] == 2 and probe["cover"] and probe["title"] == "Test Book" and probe["duration_seconds"] >= 74
        bare = make_m4b(Path(tmp) / "bare.m4b", cover=False)
        with pytest.raises(ValueError, match="no embedded cover"):
            req.probe_m4b(bare)
        cover = Path(tmp) / "book.jpg"
        folder = Path(tmp) / "delivery" / "Test Book - Test Author"
        published = req.publish(m4b, cover, folder)
        target = folder / "Test Book - Test Author.m4b"
        assert published[target.name]["sha256"] == sha256_file(m4b) == sha256_file(target)
        stamp = target.stat().st_mtime_ns
        req.publish(m4b, cover, folder)  # identical bytes are not rewritten
        assert target.stat().st_mtime_ns == stamp
        assert not list(folder.glob(".*"))
        original_bytes = target.read_bytes()
        replacement = make_m4b(Path(tmp) / "replacement.m4b", seconds=62)
        with pytest.raises(ValueError, match="refusing to replace different publication bytes"):
            req.publish(replacement, cover, folder)
        assert target.read_bytes() == original_bytes


def test_books_window() -> None:
    assert req.books_window_open(datetime(2026, 11, 24, 2, 0)) is True
    assert req.books_window_open(datetime(2026, 11, 24, 4, 59)) is True
    assert req.books_window_open(datetime(2026, 11, 24, 5, 0)) is False
    assert req.books_window_open(datetime(2026, 11, 24, 13, 0)) is False


def test_existing_books_audiobook_is_found_by_its_bytes() -> None:
    # read-only check against the machine's real Apple Books library
    assets = audiobook_assets()
    assert assets, "the Apple Books library on this machine has no audiobooks to verify against"
    asset = assets[0]
    stored = Path(asset["path"])
    files = sorted(stored.glob("*.m4b")) if stored.is_dir() else [stored]
    found = find_delivered_asset(files[0])
    assert found is not None and found["asset_id"] == asset["asset_id"]
    with tempfile.TemporaryDirectory() as tmp:
        assert find_delivered_asset(make_m4b(Path(tmp) / "never-imported.m4b", seconds=61)) is None


def test_existing_receipt_requires_matching_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        m4b = make_m4b(folder / "book.m4b")
        assert req.existing_receipt(folder, m4b) is None
        (folder / req.RECEIPT_NAME).write_text(json.dumps({"m4b_sha256": "0" * 64}), encoding="utf-8")
        assert req.existing_receipt(folder, m4b) is None


# ##################################################################
# test the series contract is recorded and published
# The registry is the durable series/character record Darren asked for. It is
# only written when a job names its series, so the request must carry that
# contract through to the worker instead of leaving it unreachable.
def test_request_records_series_contract_and_publishes_registry() -> None:
    from src.series_voices import (
        latest_series_version,
        load_series_registry,
        registry_root,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = make_config(root)
        make_epub(root / "library" / "Anne McCaffrey - Dragonsong.epub", "Dragonsong", "McCaffrey Anne")
        result = req.request(
            "Dragonsong",
            "Anne McCaffrey",
            start=False,
            config_path=config,
            series="Harper Hall of Pern",
            series_required_characters=["Menolly"],
        )
        job = result["job"]
        assert job["series"] == "Harper Hall of Pern"
        assert job["series_required_characters"] == ["Menolly"]

        # idempotent re-request keeps the recorded contract and never re-points it
        again = req.request("Dragonsong", "Anne McCaffrey", start=False, config_path=config)
        assert again["job"]["series"] == "Harper Hall of Pern"
        with pytest.raises(ValueError, match="already belongs to series"):
            req.request("Dragonsong", "Anne McCaffrey", start=False, config_path=config, series="Other Series")

        # a production with a narrator and Menolly publishes as series version 1
        project = Path(job["project"])
        project.mkdir(parents=True, exist_ok=True)
        (project / "characters.json").write_text(json.dumps({"narrator": {"name": "Narrator"}, "menolly": {"name": "Menolly"}}), encoding="utf-8")
        (project / "voices.json").write_text(
            json.dumps(
                {
                    "narrator": {"description": "A warm British narrator."},
                    "menolly": {"description": "A bright teenage girl from the sea holds."},
                }
            ),
            encoding="utf-8",
        )
        (project / "breeze_voices.json").write_text("{}", encoding="utf-8")
        from src.series_voices import publish_series

        entry = publish_series(
            Path(req.settings(config)["delivery_root"]),
            job["series"],
            project,
            character_names={"narrator": "Narrator", "menolly": "Menolly"},
            required_characters=job["series_required_characters"],
        )
        assert entry["version"] == 1
        assert "menolly" in entry["required_characters"]
        assert (registry_root(Path(req.settings(config)["delivery_root"])) / "series_registry.json").is_file()
        version, latest = latest_series_version(Path(req.settings(config)["delivery_root"]), "Harper Hall of Pern")
        assert version == 1 and latest["production"] == str(project.resolve())
        assert load_series_registry(Path(req.settings(config)["delivery_root"]))["format"] == "series-registry-v1"


# ##################################################################
# test a transient read failure does not kill the worker
# The jobs directory lives on the delivery share, where a lookup can fail
# transiently while the record is present. An absent job must still fail closed,
# so the retry is bounded and the error type is preserved.
def test_load_job_retries_a_transient_lookup_then_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = make_config(root)
        make_epub(root / "library" / "Anne McCaffrey - Dragonsong.epub", "Dragonsong", "McCaffrey Anne")
        job = req.request("Dragonsong", "Anne McCaffrey", start=False, config_path=config)["job"]
        settings = req.settings(config)
        path = req.job_path(settings, job["job_id"])

        # a present record is returned normally
        assert req.load_job(settings, job["job_id"])["job_id"] == job["job_id"]

        # a genuinely absent job still fails closed with the documented error
        with pytest.raises(FileNotFoundError, match="no audiobook job"):
            req.load_job(settings, "work-" + "0" * 16)
        with pytest.raises(FileNotFoundError, match="no audiobook job"):
            req.load_job(settings, "work-" + "1" * 16, attempts=1)

        # one transient miss is absorbed, and the real record is returned
        real_read = req.read_json
        calls = {"n": 0}

        def flaky(target: Path) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                raise FileNotFoundError(str(target))
            return real_read(target)

        req.read_json = flaky
        try:
            recovered = req.load_job(settings, job["job_id"])
        finally:
            req.read_json = real_read
        assert recovered["job_id"] == path.stem
        assert calls["n"] == 2


# ##################################################################
# test a contract recorded mid-render is honoured at publish time
# A render runs for hours. The series contract can be recorded on the job while
# it is producing, so publishing must re-read the job instead of trusting the
# copy the worker loaded when it started, or the registry never gets written.
def test_publish_reads_the_contract_recorded_during_the_render() -> None:
    import inspect

    source = inspect.getsource(req._work)
    publish_at = source.index("published = publish(")
    reload_at = source.index("load_job(config, job[\"job_id\"])")
    assert reload_at > publish_at, "the job must be re-read after publishing, before the series step"
    assert source.index('if job.get("series"):') > reload_at, "the series step must read the reloaded job"
