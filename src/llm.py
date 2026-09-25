"""Central LLM backend — local-coder category served by the arbiter GPU job server.

All book-reader LLM work (character analysis, voice descriptions, voice
mapping, script generation) goes through here. We talk to the arbiter's
OpenAI-compatible HTTP endpoint directly with stdlib urllib so there is no
heavyweight local model process and no per-call subprocess (the previous
claude-agent-sdk path spawned a ~400 MB Node CLI per call, which exhausted
local memory). The arbiter transparently farms each job to whichever Ollama
host (boringstack / darrens-mbp) is available, with failover, so this client
no longer needs to know about any single model host.

Configurable via env:
  BOOK_LLM_HOST         default http://10.0.0.254:8400  (spark arbiter server)
  BOOK_LLM_MODEL        default local-coder
  BOOK_LLM_CONCURRENCY  default 4   (parallel in-flight requests)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request

LLM_HOST = os.environ.get("BOOK_LLM_HOST", "http://10.0.0.254:8400").rstrip("/")
LLM_MODEL = os.environ.get("BOOK_LLM_MODEL", "local-coder")
MAX_CONCURRENT = int(os.environ.get("BOOK_LLM_CONCURRENCY", "4"))

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ##################################################################
# strip think
# remove any <think>...</think> reasoning block a thinking model may emit
def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


# ##################################################################
# ask sync
# one blocking chat completion against the arbiter's OpenAI-compatible
# endpoint; retries transient failures forever with backoff (the pipeline must
# never silently lose work). The arbiter handles model-host selection/failover.
def ask_sync(
    prompt: str, system: str | None = None, temperature: float = 0.2, max_tokens: int = 4096, timeout: float = 300.0
) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps(
        {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")

    attempt = 0
    while True:
        attempt += 1
        try:
            req = urllib.request.Request(
                f"{LLM_HOST}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", []) or []
            content = (choices[0].get("message", {}) if choices else {}).get("content", "") or ""
            content = strip_think(content)
            if content.strip():
                return content.strip()
            # Empty completion — retry a few times then give back empty.
            if attempt >= 5:
                return ""
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
            wait = min(5 * attempt, 60)
            print(f"  llm ({LLM_MODEL}) attempt {attempt} failed: {e} — retry in {wait}s")
            time.sleep(wait)


# ##################################################################
# semaphore
# bound concurrent in-flight requests so we don't swamp the single model.
# Keyed per running event loop: the pipeline runs each step under its own
# asyncio.run(), and an asyncio.Semaphore is bound to the loop it was created
# in — a module-level singleton would raise "bound to a different event loop"
# on the second step.
_sems: dict = {}


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _sems.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(MAX_CONCURRENT)
        _sems[loop] = sem
    return sem


# ##################################################################
# ask
# async chat completion — runs the blocking call in a worker thread
async def ask(prompt: str, system: str | None = None, temperature: float = 0.2, max_tokens: int = 4096) -> str:
    async with _semaphore():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: ask_sync(prompt, system, temperature, max_tokens),
        )
