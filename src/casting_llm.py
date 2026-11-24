"""Content-addressed record/replay cache for casting LLM requests.

Doctrine ("real but fast"): every cached byte came from the real model. A miss
performs a live bounded call through src.llm and records the answer; a hit
replays the recorded real answer with zero network. Recordings are
human-readable JSON, one file per request key.

Cache activation is explicit: callers pass ``cache_root`` (tests point it at a
committed recordings directory). Production pipeline paths never pass it, so
prod is always live — a cache-layer failure can never break the live call.

Key = sha256 over the canonical serialization of the FULL router backend
configuration (endpoint URLs, models, styles, ctx, think — everything that
changes which model answers and how) plus prompt, temperature, max_tokens.
The recorded entry also stores the route model that actually answered, so a
recorded run is inspectable. A corrupt/partial entry is a miss that self-heals
on rewrite.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

from src import llm

ENTRY_VERSION = 2


# ##################################################################
# cache identity
def router_identity() -> dict:
    """Full routing identity: both backends' complete configuration."""
    return {
        "primary": dataclasses.asdict(llm.ROUTER_CONFIG.primary),
        "backup": dataclasses.asdict(llm.ROUTER_CONFIG.backup),
    }


def _key(prompt: str, temperature: float, max_tokens: int) -> str:
    identity = {
        "router": router_identity(),
        "prompt": prompt,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _read_entry(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != ENTRY_VERSION or not isinstance(data.get("response"), str):
        return None
    return data


def _write_entry(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # unique temp name: parallel writers of the same key never collide
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ##################################################################
# cached sync ask
# bounded live attempts; record on miss (with the route that answered),
# replay on hit
def ask_cached(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    root: Path | str | None = None,
    max_attempts: int | None = None,
) -> str:
    if root is None:
        return llm.ask_sync(
            prompt, temperature=temperature, max_tokens=max_tokens, max_attempts=max_attempts
        )
    path = Path(root) / f"{_key(prompt, temperature, max_tokens)}.json"
    entry = _read_entry(path)
    if entry is not None:
        return entry["response"]
    routing_events: list[str] = []
    response = llm.ask_sync(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        max_attempts=max_attempts,
        routing_events=routing_events,
    )
    _write_entry(
        path,
        {
            "version": ENTRY_VERSION,
            "router": router_identity(),
            "routes_used": routing_events,
            "request": {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens},
            "response": response,
        },
    )
    return response


# ##################################################################
# cached async ask
# same cache, executor delegation mirrors src.llm.ask_async
async def ask_cached_async(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    root: Path | str | None = None,
    max_attempts: int | None = None,
) -> str:
    loop = asyncio.get_running_loop()
    async with llm._semaphore():
        return await loop.run_in_executor(
            None,
            lambda: ask_cached(
                prompt, temperature=temperature, max_tokens=max_tokens, root=root, max_attempts=max_attempts
            ),
        )
