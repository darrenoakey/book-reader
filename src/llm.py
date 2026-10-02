"""Central, ping-gated Book Reader LLM routing.

The Boringstack is always the primary when it answers ICMP.  The local MacBook
Ollama is a backup only while that ICMP probe fails; HTTP failures from a
reachable primary never cause a backend switch.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tomllib

PRIMARY_DEFAULT_URL = "http://10.0.0.42:11434"
PRIMARY_DEFAULT_MODEL = "qwen3.6:35b-a3b"
BACKUP_DEFAULT_URL = "http://127.0.0.1:11434"
BACKUP_DEFAULT_MODEL = "qwen3:8b"
PERMANENT_HTTP_ATTEMPTS = 2
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Backend:
    url: str
    ping_host: str
    model: str
    style: str
    num_ctx: int
    think: bool
    expected_identity: str | None = None


@dataclass(frozen=True)
class RouterConfig:
    primary: Backend
    backup: Backend
    concurrency: int


# ##################################################################
# load legacy config
# retain the original tuple API for callers and existing local configuration while routing uses the richer configuration below.
def load_llm_config(path: Path) -> tuple[str, str, int, str]:
    values: dict = {}
    if path.is_file():
        with path.open("rb") as stream:
            values = tomllib.load(stream).get("llm", {})
    host = str(values.get("host", PRIMARY_DEFAULT_URL)).rstrip("/")
    model = str(values.get("model", PRIMARY_DEFAULT_MODEL))
    concurrency = int(values.get("concurrency", 2))
    style = str(values.get("style", "ollama")).lower()
    if not host.startswith(("http://", "https://")) or concurrency < 1 or style not in {"openai", "ollama"}:
        raise ValueError("local/config.toml [llm] has invalid host, concurrency, or style")
    return host, model, concurrency, style


# ##################################################################
# load router config
# make both backend payloads explicit; old host/model settings remain the primary until local config is migrated.
def load_router_config(path: Path) -> RouterConfig:
    legacy_host, legacy_model, concurrency, _legacy_style = load_llm_config(path)
    values: dict = {}
    if path.is_file():
        with path.open("rb") as stream:
            values = tomllib.load(stream).get("llm", {})

    def backend(name: str, default_url: str, default_host: str, default_model: str, default_ctx: int) -> Backend:
        url = str(values.get(f"{name}_url", default_url)).rstrip("/")
        ping_host = str(values.get(f"{name}_ping_host", default_host))
        model = str(values.get(f"{name}_model", default_model))
        style = str(values.get(f"{name}_style", "ollama")).lower()
        num_ctx = int(values.get(f"{name}_num_ctx", default_ctx))
        think = bool(values.get(f"{name}_think", False))
        identity = values.get(f"{name}_identity")
        if not url.startswith(("http://", "https://")) or not ping_host or not model or style not in {"openai", "ollama"} or num_ctx < 1024:
            raise ValueError(f"local/config.toml [llm] has invalid {name} backend")
        return Backend(url, ping_host, model, style, num_ctx, think, str(identity) if identity else None)

    primary = backend("primary", legacy_host, "10.0.0.42", legacy_model, 32768)
    backup = backend("backup", BACKUP_DEFAULT_URL, "127.0.0.1", BACKUP_DEFAULT_MODEL, 40960)
    return RouterConfig(primary, backup, concurrency)


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "local" / "config.toml"
ROUTER_CONFIG = load_router_config(_CONFIG_PATH)
# Legacy exports remain for existing imports. They describe primary, not a sticky selected backend.
LLM_HOST, LLM_MODEL, MAX_CONCURRENT, LLM_STYLE = load_llm_config(_CONFIG_PATH)


# ##################################################################
# ping host
# use bounded ICMP only; a failed or malformed probe is fail-closed and is never inferred from an HTTP outcome.
def ping_host(host: str) -> bool:
    try:
        result = subprocess.run(["/sbin/ping", "-c", "1", "-W", "500", host], capture_output=True, timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


# ##################################################################
# local identity
# prevent a loopback backup setting from silently using the wrong Mac when an expected identity was configured.
def backup_identity_matches(backend: Backend) -> bool:
    if not backend.expected_identity:
        return True
    if backend.ping_host not in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        result = subprocess.run(["/usr/sbin/scutil", "--get", "LocalHostName"], capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == backend.expected_identity


# ##################################################################
# select backend
# choose afresh before every request attempt so the next retry returns to primary the instant its ICMP recovers.
def select_backend(config: RouterConfig = ROUTER_CONFIG) -> Backend:
    if ping_host(config.primary.ping_host):
        return config.primary
    if ping_host(config.backup.ping_host) and backup_identity_matches(config.backup):
        return config.backup
    raise RuntimeError("Book Reader LLM unavailable: primary ICMP failed and verified backup is unavailable")


# ##################################################################
# strip think
# remove thinking markup if a native Ollama model emits it despite think=false.
def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


# ##################################################################
# request payload
# rebuild from the selected backend on every retry so model/style/context never leak from a previous route.
def request_for(backend: Backend, messages: list[dict], temperature: float, max_tokens: int, response_schema: dict | None) -> tuple[str, bytes]:
    if backend.style == "ollama":
        request: dict = {"model": backend.model, "messages": messages, "think": backend.think, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": backend.num_ctx}}
        if response_schema is not None:
            request["format"] = response_schema
        return f"{backend.url}/api/chat", json.dumps(request).encode("utf-8")
    request = {"model": backend.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if response_schema is not None:
        request["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "strict": True, "schema": response_schema}}
    return f"{backend.url}/v1/chat/completions", json.dumps(request).encode("utf-8")


# ##################################################################
# ask sync
# make a central native request; only a fresh ICMP result may select backup, while permanent HTTP errors stop after a bounded number of attempts.
def ask_sync(prompt: str, system: str | None = None, temperature: float = 0.2, max_tokens: int = 4096, timeout: float = 300.0, response_schema: dict | None = None, *, config: RouterConfig = ROUTER_CONFIG, max_attempts: int | None = None, routing_events: list[str] | None = None) -> str:
    messages: list[dict] = ([] if not system else [{"role": "system", "content": system}]) + [{"role": "user", "content": prompt}]
    attempt = permanent_failures = 0
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        backend = select_backend(config)
        if routing_events is not None:
            routing_events.append(backend.model)
        url, payload = request_for(backend, messages, temperature, max_tokens, response_schema)
        try:
            request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = (data.get("message") or {}).get("content", "") if backend.style == "ollama" else ((data.get("choices") or [{}])[0].get("message", {}).get("content", ""))
            content = strip_think(content or "")
            if content:
                return content
            raise RuntimeError("empty completion")
        except urllib.error.HTTPError as error:
            if 400 <= error.code < 500:
                permanent_failures += 1
                if permanent_failures >= PERMANENT_HTTP_ATTEMPTS:
                    raise RuntimeError(f"LLM {backend.model} returned permanent HTTP {error.code} after {attempt} attempts") from error
            print(f"  llm ({backend.model}) HTTP {error.code}; retrying selected route")
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, RuntimeError) as error:
            print(f"  llm ({backend.model}) attempt {attempt} failed: {error}; retrying selected route")
        if max_attempts is not None and attempt >= max_attempts:
            break
        time.sleep(min(attempt, 5))
    raise RuntimeError(f"LLM request exhausted {attempt} attempts")


_sems: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    return _sems.setdefault(loop, asyncio.Semaphore(MAX_CONCURRENT))


# ##################################################################
# ask async
# preserve central routing for all coroutine callers by delegating exactly to ask_sync in a worker thread.
async def ask_async(prompt: str, system: str | None = None, temperature: float = 0.2, max_tokens: int = 4096, response_schema: dict | None = None) -> str:
    async with _semaphore():
        return await asyncio.get_running_loop().run_in_executor(None, lambda: ask_sync(prompt, system, temperature, max_tokens, response_schema=response_schema))


# Backward-compatible public name for pipeline modules already importing ask.
ask = ask_async
