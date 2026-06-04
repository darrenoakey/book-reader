import logging
import os
import time
from pathlib import Path

from arbiter_client import ArbiterClient, ArbiterError

log = logging.getLogger(__name__)

MAX_RETRIES = 100000
RETRY_BACKOFF_SEC = 5

# Talk to the arbiter server on spark directly. The default ArbiterClient base
# is a local proxy (localhost:8399) that isn't always running; for TTS we only
# need submit/poll/result and (with force=True) results come back inline as
# base64, so a direct connection needs no local mount/proxy service.
ARBITER_BASE = os.environ.get("ARBITER_BASE", "http://10.0.0.254:8400")


def _client(timeout: float = 60) -> ArbiterClient:
    return ArbiterClient(base_url=ARBITER_BASE, timeout=timeout)


# ##################################################################
# submit
# submit a job of given type and return its id, retrying on transient errors
def _submit(client: ArbiterClient, job_type: str, params: dict) -> str:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.submit(job_type, **params)
        except (ArbiterError, ConnectionError, OSError) as e:
            last_err = e
            wait = min(RETRY_BACKOFF_SEC * (attempt + 1), 60)
            log.warning("submit %s attempt %d/%d failed: %s — retrying in %ds",
                        job_type, attempt + 1, MAX_RETRIES, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"submit {job_type} failed after {MAX_RETRIES} attempts: {last_err}")


# ##################################################################
# fetch
# poll an existing job and write result; on failure resubmit and retry
def _fetch(client: ArbiterClient, job_id: str, job_type: str, params: dict,
           output_path: Path) -> None:
    """Wait for a job to finish. NEVER resubmit on a mere poll timeout — that
    just pushes the job to the back of the queue and makes things worse.
    Only resubmit if the job is genuinely terminated (failed or cancelled)."""
    current_jid = job_id
    while True:
        try:
            # poll with effectively infinite timeout — keep waiting
            client.poll(current_jid, interval=2.0, timeout=31536000)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(client.get_result_bytes(current_jid))
            if output_path.stat().st_size >= 100:
                return
            log.warning("fetch %s for %s returned empty — resubmitting",
                        job_type, output_path.name)
            current_jid = _submit(client, job_type, params)
        except ArbiterError as e:
            msg = str(e).lower()
            if "failed" in msg or "cancelled" in msg or "timed out" in msg:
                # job is dead — resubmit a new one
                log.warning("fetch %s for %s: job died (%s) — resubmitting",
                            job_type, output_path.name, e)
                current_jid = _submit(client, job_type, params)
            else:
                # transient connection error — keep polling the same job
                log.warning("fetch %s for %s: transient (%s) — retrying poll",
                            job_type, output_path.name, e)
                time.sleep(5)
        except (ConnectionError, OSError) as e:
            log.warning("fetch %s for %s: connection (%s) — retrying poll",
                        job_type, output_path.name, e)
            time.sleep(5)


# ##################################################################
# tts design to file
# generate a voice from description and save the wav locally
def tts_design_to_file(description: str, text: str, output_path: Path,
                       language: str = "English", temperature: float = 0.9) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 100:
        return output_path
    client = _client(60)
    params = {
        "text": text,
        "instruct": description,
        "language": language,
        "temperature": temperature,
        "force": True,
    }
    jid = _submit(client, "tts-design", params)
    _fetch(client, jid, "tts-design", params, output_path)
    return output_path


# ##################################################################
# tts clone to file
# clone a voice using a pre-registered speaker_id (no ref audio sent)
def tts_clone_to_file(speaker_id: str, text: str, output_path: Path,
                      language: str = "English", temperature: float = 0.7) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 100:
        return output_path
    client = _client(120)
    params = {
        "text": text,
        "speaker_id": speaker_id,
        "language": language,
        "temperature": temperature,
        "force": True,
    }
    jid = _submit(client, "tts-clone", params)
    _fetch(client, jid, "tts-clone", params, output_path)
    return output_path


# ##################################################################
# tts clone many
# submit a batch of tts-clone jobs by speaker_id only — ref audio is
# pre-registered on spark, jobs never carry ref bytes
IN_FLIGHT_WINDOW = 32


def tts_clone_many(jobs: list[dict], ref_audio_path: Path,
                   language: str = "English",
                   temperature: float = 0.7) -> list[Path]:
    """Submit a batch of tts-clone jobs all sharing one staged ref WAV,
    keeping ~IN_FLIGHT_WINDOW jobs in flight at a time using a thread pool
    so arbiter's worker slots stay saturated."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from arbiter_client import stage_file
    spark_path = stage_file(ref_audio_path, keep_for_seconds=6 * 86400)

    todo = [j for j in jobs
            if not (j["output_path"].exists() and j["output_path"].stat().st_size >= 100)]
    if not todo:
        return [j["output_path"] for j in jobs]

    def _do_one(j: dict) -> None:
        client = _client(120)
        params = {
            "text": j["text"],
            "ref_audio_file": spark_path,
            "language": language,
            "temperature": temperature,
            "force": True,
        }
        jid = _submit(client, "tts-clone", params)
        _fetch(client, jid, "tts-clone", params, j["output_path"])

    print(f"  {len(todo)} jobs, window={IN_FLIGHT_WINDOW}")
    done_count = 0
    with ThreadPoolExecutor(max_workers=IN_FLIGHT_WINDOW) as pool:
        futures = {pool.submit(_do_one, j): j for j in todo}
        for fut in as_completed(futures):
            fut.result()  # propagate exceptions
            done_count += 1
            if done_count % 100 == 0:
                print(f"    {done_count}/{len(todo)} done")

    return [j["output_path"] for j in jobs]


# ##################################################################
# tts kokoro many
# submit a batch of kokoro TTS jobs (one per line) in parallel. Each job dict
# carries its own kokoro voice spec + speed, so different speakers render with
# different voices in the same batch. Kokoro is tiny+fast, so a wide window
# keeps the single spark worker saturated.
def tts_kokoro_many(jobs: list[dict]) -> list[Path]:
    """Each job: {text, voice, speed, output_path}. Fills output_path WAVs."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    todo = [j for j in jobs
            if not (j["output_path"].exists() and j["output_path"].stat().st_size >= 100)]
    if not todo:
        return [j["output_path"] for j in jobs]

    def _do_one(j: dict) -> None:
        client = _client(120)
        params = {
            "text": j["text"],
            "voice": j.get("voice", "af_heart"),
            "speed": float(j.get("speed", 1.0)),
            "force": True,
        }
        jid = _submit(client, "tts-kokoro", params)
        _fetch(client, jid, "tts-kokoro", params, j["output_path"])

    print(f"  {len(todo)} kokoro jobs, window={IN_FLIGHT_WINDOW}")
    done_count = 0
    with ThreadPoolExecutor(max_workers=IN_FLIGHT_WINDOW) as pool:
        futures = {pool.submit(_do_one, j): j for j in todo}
        for fut in as_completed(futures):
            fut.result()
            done_count += 1
            if done_count % 100 == 0:
                print(f"    {done_count}/{len(todo)} done")

    return [j["output_path"] for j in jobs]


__all__ = [
    "tts_design_to_file", "tts_clone_to_file", "tts_clone_many",
    "tts_kokoro_many",
]
