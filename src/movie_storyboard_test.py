"""Tests for movie_storyboard: per-book scene-seconds image frequency."""

from pathlib import Path


# ##################################################################
# test choose scene seconds
# scene_seconds.txt overrides the default; garbage and out-of-range fail closed
def test_choose_scene_seconds() -> None:
    import tempfile

    from src.movie_storyboard import TARGET_SECONDS, choose_scene_seconds

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        assert choose_scene_seconds(out) == TARGET_SECONDS
        (out / "scene_seconds.txt").write_text("15")
        assert choose_scene_seconds(out) == 15.0
        (out / "scene_seconds.txt").write_text("banana")
        try:
            choose_scene_seconds(out)
            raise AssertionError("garbage scene_seconds must raise")
        except ValueError:
            pass
        (out / "scene_seconds.txt").write_text("999")
        try:
            choose_scene_seconds(out)
            raise AssertionError("out-of-range scene_seconds must raise")
        except ValueError:
            pass
