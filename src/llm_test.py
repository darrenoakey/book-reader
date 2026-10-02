"""Real native-Ollama tests for Book Reader's ping-gated LLM router."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.llm import (
    Backend,
    RouterConfig,
    ask_async,
    ask_sync,
    load_llm_config,
    load_router_config,
    request_for,
    select_backend,
    strip_think,
)

BACKUP = Backend("http://127.0.0.1:11434", "127.0.0.1", "qwen3:8b", "ollama", 40960, False)


# ##################################################################
# test local TOML migration
# retain the legacy tuple while accepting explicit per-backend route settings and context limits.
def test_load_router_config() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.toml"
        path.write_text(
            "[llm]\nhost = 'http://10.0.0.42:11434'\nmodel = 'qwen3.6:35b-a3b'\nconcurrency = 2\nstyle = 'ollama'\n"
            "primary_url = 'http://10.0.0.42:11434'\nprimary_ping_host = '10.0.0.42'\nprimary_model = 'qwen3.6:35b-a3b'\nprimary_num_ctx = 32768\n"
            "backup_url = 'http://127.0.0.1:11434'\nbackup_ping_host = '127.0.0.1'\nbackup_model = 'qwen3:8b'\nbackup_num_ctx = 40960\n",
            encoding="utf-8",
        )
        assert load_llm_config(path) == ("http://10.0.0.42:11434", "qwen3.6:35b-a3b", 2, "ollama")
        config = load_router_config(path)
        assert config.primary.num_ctx == 32768
        assert config.backup.num_ctx == 40960


# ##################################################################
# test live backup schema chat
# prove the verified loopback native Ollama model accepts strict JSON schema and the full configured context payload.
def test_backup_native_schema_chat() -> None:
    config = RouterConfig(Backend("http://203.0.113.1:11434", "203.0.113.1", "absent", "ollama", 32768, False), BACKUP, 1)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    response = ask_sync("Reply with the JSON object {\"ok\":true} only.", response_schema=schema, max_tokens=64, timeout=90, config=config, max_attempts=1)
    assert response == '{"ok":true}'


# ##################################################################
# test pingable primary http 404 never switches
# a live loopback primary that returns model-not-found stays selected; backup availability cannot override a successful ICMP probe.
def test_pingable_primary_http_404_does_not_fallback() -> None:
    primary = Backend("http://127.0.0.1:11434", "127.0.0.1", "book-reader-intentionally-missing-model", "ollama", 40960, False)
    events: list[str] = []
    with pytest.raises(RuntimeError, match="permanent HTTP 404"):
        ask_sync("hello", config=RouterConfig(primary, BACKUP, 1), timeout=10, max_attempts=2, routing_events=events)
    assert events == [primary.model, primary.model]


# ##################################################################
# test pingable primary connection failure never switches
# a closed local OS port is a real reachable-primary transport failure, not permission to use backup.
def test_pingable_primary_closed_port_does_not_fallback() -> None:
    primary = Backend("http://127.0.0.1:9", "127.0.0.1", "qwen3:8b", "ollama", 40960, False)
    events: list[str] = []
    with pytest.raises(RuntimeError, match="exhausted"):
        ask_sync("hello", config=RouterConfig(primary, BACKUP, 1), timeout=1, max_attempts=2, routing_events=events)
    assert events == [primary.model, primary.model]


# ##################################################################
# test fresh selection policy
# primary selection is stateless: each call asks ICMP again rather than retaining a previous backup result.
def test_select_backend_primary_loopback() -> None:
    primary = Backend("http://127.0.0.1:11434", "127.0.0.1", "qwen3:8b", "ollama", 40960, False)
    assert select_backend(RouterConfig(primary, BACKUP, 1)) == primary


# ##################################################################
# test payload context
# native requests carry the selected backend model and bounded context on every independently-built payload.
def test_request_payload_uses_selected_backend() -> None:
    url, payload = request_for(BACKUP, [{"role": "user", "content": "hello"}], 0.2, 64, None)
    assert url == "http://127.0.0.1:11434/api/chat"
    assert b'"model": "qwen3:8b"' in payload and b'"num_ctx": 40960' in payload


# ##################################################################
# test strip think
# reasoning blocks a thinking model emits must be removed.
def test_strip_think() -> None:
    assert strip_think("<think>reasoning</think>Answer") == "Answer"


# ##################################################################
# test async central route
# coroutine callers must flow through ask_sync and the same live ping-gated native backup.
def test_ask_async_real() -> None:
    async def go() -> str:
        return await ask_async("Reply with exactly one word: PONG", max_tokens=32)

    assert "pong" in asyncio.run(go()).lower()
