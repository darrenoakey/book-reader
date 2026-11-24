import asyncio
import json
from pathlib import Path

from src import llm
from src.casting_llm import _key, _read_entry, ask_cached, ask_cached_async

RECORDINGS = Path(__file__).parent / "testdata" / "casting-recordings"
PROMPT = "For an audiobook casting connectivity check, reply only with the word ready."


# ##################################################################
# real content-addressed transport
# A recording is created by the configured real backend, then copied into an
# isolated cache. Corruption causes a real re-query, never a fabricated reply.
def test_real_cache_replay_and_corruption(tmp_path: Path) -> None:
    response = ask_cached(PROMPT, root=RECORDINGS, max_tokens=32)
    key = _key(PROMPT, 0.2, 32)
    recorded = RECORDINGS / f"{key}.json"
    entry = json.loads(recorded.read_text())
    assert response and entry["routes_used"]
    local = tmp_path / recorded.name
    local.write_bytes(recorded.read_bytes())
    modified = local.stat().st_mtime_ns
    assert ask_cached(PROMPT, root=tmp_path, max_tokens=32) == response
    assert local.stat().st_mtime_ns == modified
    local.write_text("truncated")
    assert _read_entry(local) is None
    assert ask_cached(PROMPT, root=tmp_path, max_tokens=32)
    assert _read_entry(local)["routes_used"]


def test_async_cache_uses_real_recording() -> None:
    result = asyncio.run(ask_cached_async(PROMPT, root=RECORDINGS, max_tokens=32))
    assert result
    assert _key(PROMPT, 0.2, 32) != _key(PROMPT, 0.2, 33)


def test_live_casting_uses_the_router_until_a_real_answer(monkeypatch) -> None:
    calls = []

    def answer(prompt, **kwargs):
        calls.append(kwargs)
        return "real response"

    monkeypatch.setattr(llm, "ask_sync", answer)
    assert ask_cached("live voice description") == "real response"
    assert len(calls) == 1
    assert calls[0]["max_attempts"] is None
