import json
from datetime import datetime
from pathlib import Path


# ##################################################################
# append state
# add an entry to the state.jsonl log
def append_state(output_dir: Path, step: str, detail: str) -> None:
    state_path = output_dir / "state.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "detail": detail,
    }
    with open(state_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ##################################################################
# is step complete
# check if a step has been completed in the state log
def is_step_complete(output_dir: Path, step: str) -> bool:
    state_path = output_dir / "state.jsonl"
    if not state_path.exists():
        return False
    with open(state_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("step") == step and entry.get("detail") == "complete":
                    return True
    return False


# ##################################################################
# mark step complete
# mark a step as completed in the state log
def mark_step_complete(output_dir: Path, step: str) -> None:
    append_state(output_dir, step, "complete")


# ##################################################################
# get completed items
# get list of completed items for a step (for incremental progress)
def get_completed_items(output_dir: Path, step: str) -> set[str]:
    state_path = output_dir / "state.jsonl"
    completed = set()
    if not state_path.exists():
        return completed
    with open(state_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry.get("step") == step and entry.get("detail") != "complete":
                    completed.add(entry.get("detail"))
    return completed


# ##################################################################
# mark item done
# mark a specific item as done within a step
def mark_item_done(output_dir: Path, step: str, item: str) -> None:
    append_state(output_dir, step, item)
