"""Real integration tests for the arbiter-backed LLM client.

These call the LIVE arbiter GPU job server (local-coder at 10.0.0.254:8400).
No mocks — a tiny completion is exercised end to end so a broken endpoint or a
changed request/response shape fails the suite immediately.
"""

import asyncio

from src.llm import LLM_HOST, LLM_MODEL, ask, ask_sync, strip_think


# ##################################################################
# test config points at arbiter
# the default backend must be the arbiter OpenAI endpoint, not raw Ollama
def test_config_targets_arbiter() -> None:
    assert LLM_MODEL == "local-coder"
    assert LLM_HOST.endswith(":8400")


# ##################################################################
# test strip think
# reasoning blocks a thinking model emits must be removed
def test_strip_think() -> None:
    assert strip_think("<think>reasoning</think>Answer") == "Answer"
    assert strip_think("plain") == "plain"


# ##################################################################
# test ask sync real completion
# one tiny blocking completion through the live arbiter server
def test_ask_sync_real() -> None:
    out = ask_sync(
        "Reply with exactly one word and nothing else: PONG",
        system="You are a terse test fixture. Output only what is asked.",
        max_tokens=200,
        timeout=180.0,
    )
    assert out.strip(), "arbiter returned empty content"
    assert "pong" in out.lower(), f"unexpected completion: {out!r}"


# ##################################################################
# test ask async real completion
# the async wrapper must also complete a real call through the arbiter
def test_ask_async_real() -> None:
    async def _go() -> str:
        return await ask(
            "Reply with exactly one word and nothing else: PONG",
            system="You are a terse test fixture. Output only what is asked.",
            max_tokens=200,
        )

    out = asyncio.run(_go())
    assert "pong" in out.lower(), f"unexpected completion: {out!r}"
