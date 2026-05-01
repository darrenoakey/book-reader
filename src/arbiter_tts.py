import logging
import time
from pathlib import Path

from arbiter_client import ArbiterClient, ArbiterError

log = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 30


# ##################################################################
# submit and fetch one
# submit a tts-design job, poll, write result to output_path with retries
def _submit_and_fetch(client: ArbiterClient, description: str, text: str,
                      output_path: Path, language: str, temperature: float) -> None:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            jid = client.submit(
                "tts-design",
                text=text,
                instruct=description,
                language=language,
                temperature=temperature,
                force=True,
            )
            client.poll(jid, interval=1.0, timeout=900)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(client.get_result_bytes(jid))
            if output_path.stat().st_size >= 100:
                return
            raise RuntimeError("empty output")
        except (ArbiterError, RuntimeError, ConnectionError, OSError) as e:
            last_err = e
            wait = RETRY_BACKOFF_SEC * (attempt + 1)
            log.warning("tts-design attempt %d/%d for %s failed: %s — retrying in %ds",
                        attempt + 1, MAX_RETRIES, output_path.name, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"tts-design failed after {MAX_RETRIES} attempts for {output_path}: {last_err}")


# ##################################################################
# tts design to file
# generate a voice from description and save the wav locally
def tts_design_to_file(description: str, text: str, output_path: Path,
                       language: str = "English", temperature: float = 0.9) -> Path:
    client = ArbiterClient(timeout=60)
    _submit_and_fetch(client, description, text, output_path, language, temperature)
    return output_path


# ##################################################################
# tts design many
# submit tts-design jobs in parallel batches with per-job retries
def tts_design_many(jobs: list[dict], language: str = "English",
                    temperature: float = 0.9) -> list[Path]:
    client = ArbiterClient(timeout=60)
    submissions: list[dict] = []
    # Submit all jobs first; on submit failure, retry inline.
    for j in jobs:
        for attempt in range(MAX_RETRIES):
            try:
                jid = client.submit(
                    "tts-design",
                    text=j["text"],
                    instruct=j["description"],
                    language=language,
                    temperature=temperature,
                    force=True,
                )
                submissions.append({
                    "job_id": jid,
                    "output_path": j["output_path"],
                    "description": j["description"],
                    "text": j["text"],
                })
                break
            except (ArbiterError, ConnectionError, OSError) as e:
                wait = RETRY_BACKOFF_SEC * (attempt + 1)
                log.warning("submit attempt %d/%d failed: %s — retrying in %ds",
                            attempt + 1, MAX_RETRIES, e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"could not submit job for {j['output_path']}")

    results: list[Path] = []
    for sub in submissions:
        out_path: Path = sub["output_path"]
        try:
            client.poll(sub["job_id"], interval=0.5, timeout=900)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(client.get_result_bytes(sub["job_id"]))
            if out_path.stat().st_size < 100:
                raise RuntimeError("empty output")
        except (ArbiterError, RuntimeError, ConnectionError, OSError) as e:
            # Job lost (arbiter restart, etc) — re-run synchronously with full retries.
            log.warning("poll for %s failed (%s) — resubmitting with retries", out_path.name, e)
            _submit_and_fetch(client, sub["description"], sub["text"], out_path, language, temperature)
        results.append(out_path)
    return results


__all__ = ["tts_design_to_file", "tts_design_many"]
