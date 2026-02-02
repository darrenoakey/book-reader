import tempfile
from pathlib import Path

from src.state import (
    get_completed_items,
    is_step_complete,
    mark_item_done,
    mark_step_complete,
)


# ##################################################################
# test append and check state
# verify state tracking works
def test_state_tracking() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        assert not is_step_complete(tmpdir, "extract")
        mark_step_complete(tmpdir, "extract")
        assert is_step_complete(tmpdir, "extract")
        assert not is_step_complete(tmpdir, "other_step")


# ##################################################################
# test incremental items
# verify tracking individual items within a step
def test_incremental_items() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        assert get_completed_items(tmpdir, "voices") == set()
        mark_item_done(tmpdir, "voices", "narrator")
        mark_item_done(tmpdir, "voices", "john")
        completed = get_completed_items(tmpdir, "voices")
        assert "narrator" in completed
        assert "john" in completed
        assert "mary" not in completed
