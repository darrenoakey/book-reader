"""Real native-Ollama tests for Book Reader's ping-gated LLM router."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from src.llm import (
    Backend,
    RouterConfig,
    ask_async,
    ask_sync,
    classify_ping_result,
    load_llm_config,
    load_router_config,
    ping_host,
    request_for,
    select_backend,
    strip_think,
    telemetry_scope,
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
# test real unreachable ping
# a genuine ICMP nonresponse is the only normal condition that permits selecting backup.
def test_ping_host_real_unreachable_is_false() -> None:
    assert ping_host("203.0.113.1") is False


# ##################################################################
# test probe denial classification
# pure probe-result policy must fail closed on local denial or malformed output rather than silently choosing backup.
def test_ping_denial_and_invalid_probe_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="denied for target primary"):
        classify_ping_result(2, "ping: sendto: Operation not permitted", "primary")
    with pytest.raises(RuntimeError, match="invalid for target primary"):
        classify_ping_result(64, "usage: ping", "primary")
    assert classify_ping_result(2, "1 packets transmitted, 0 packets received, 100.0% packet loss", "primary") is False


# ##################################################################
# test live backup schema chat
# prove the verified loopback native Ollama model accepts strict JSON schema and the full configured context payload.
def test_backup_native_schema_chat() -> None:
    config = RouterConfig(
        Backend("http://203.0.113.1:11434", "203.0.113.1", "absent", "ollama", 32768, False), BACKUP, 1
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    response = ask_sync(
        'Reply with the JSON object {"ok":true} only.',
        response_schema=schema,
        max_tokens=64,
        timeout=90,
        config=config,
        max_attempts=1,
    )
    assert response == '{"ok":true}'


# ##################################################################
# test live native telemetry
# use the real backup response to prove stdout diagnostics carry only numeric/provider metadata and no source or prompt text.
def test_live_native_telemetry_is_metadata_only(capsys: pytest.CaptureFixture[str]) -> None:
    config = RouterConfig(
        Backend("http://203.0.113.1:11434", "203.0.113.1", "absent", "ollama", 32768, False), BACKUP, 1
    )
    prompt = 'Reply with the JSON object {"ok":true} only. Do not repeat this diagnostic sentinel: CAST_SOURCE_SECRET.'
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    with telemetry_scope("native_telemetry_test", 7, 8, 0):
        assert ask_sync(prompt, response_schema=schema, max_tokens=64, timeout=90, config=config, max_attempts=1) == '{"ok":true}'
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    call = next(event for event in events if event["event"] == "book_reader_llm_call")
    phase = next(event for event in events if event["event"] == "book_reader_llm_phase")
    assert call["phase"] == "native_telemetry_test"
    assert call["batch_start"] == 7 and call["batch_end"] == 8 and call["phase_call"] == 1
    assert call["model"] == BACKUP.model and call["think_requested"] is False
    assert call["input_chars"] == len(prompt) and call["output_chars"] == len('{"ok":true}')
    assert call["total_duration_ns"] is not None and call["prompt_eval_count"] is not None and call["eval_count"] is not None
    assert call["thinking_tokens_reported"] is None
    assert "CAST_SOURCE_SECRET" not in json.dumps(call)
    assert phase["calls_started"] == 1 and phase["outcome"] == "ok" and phase["wall_duration_ns"] > 0


# ##################################################################
# test pingable primary http 404 never switches
# a live loopback primary that returns model-not-found stays selected; backup availability cannot override a successful ICMP probe.
def test_pingable_primary_http_404_does_not_fallback() -> None:
    primary = Backend(
        "http://127.0.0.1:11434", "127.0.0.1", "book-reader-intentionally-missing-model", "ollama", 40960, False
    )
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
