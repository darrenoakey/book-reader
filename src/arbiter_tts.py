import base64
from pathlib import Path

from arbiter_client import ArbiterClient


# ##################################################################
# tts design to file
# generate a voice from description and save the wav locally
def tts_design_to_file(description: str, sample_text: str, output_path: Path,
                       language: str = "English", temperature: float = 0.9) -> Path:
    client = ArbiterClient(timeout=60)
    job_id = client.submit(
        "tts-design",
        text=sample_text,
        instruct=description,
        language=language,
        temperature=temperature,
        force=True,
    )
    client.poll(job_id, interval=1.0, timeout=900)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(client.get_result_bytes(job_id))
    if output_path.stat().st_size < 100:
        raise RuntimeError(f"tts-design produced empty output for {output_path}")
    return output_path


# ##################################################################
# tts clone to file
# clone a voice from a local reference wav and save the result locally
def tts_clone_to_file(ref_wav: Path, text: str, output_path: Path,
                      language: str = "English", temperature: float = 0.3) -> Path:
    client = ArbiterClient(timeout=120)
    ref_b64 = base64.b64encode(ref_wav.read_bytes()).decode()
    job_id = client.submit(
        "tts-clone",
        text=text,
        ref_audio=ref_b64,
        language=language,
        temperature=temperature,
        force=True,
    )
    client.poll(job_id, interval=1.0, timeout=1200)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(client.get_result_bytes(job_id))
    if output_path.stat().st_size < 100:
        raise RuntimeError(f"tts-clone produced empty output for {output_path}")
    return output_path


# ##################################################################
# tts clone many
# submit tts-clone jobs in parallel and write results sequentially
def tts_clone_many(jobs: list[dict], language: str = "English",
                   temperature: float = 0.3) -> list[Path]:
    client = ArbiterClient(timeout=120)
    submissions: list[dict] = []
    ref_cache: dict[Path, str] = {}
    for j in jobs:
        ref_wav: Path = j["ref_wav"]
        if ref_wav not in ref_cache:
            ref_cache[ref_wav] = base64.b64encode(ref_wav.read_bytes()).decode()
        jid = client.submit(
            "tts-clone",
            text=j["text"],
            ref_audio=ref_cache[ref_wav],
            language=language,
            temperature=temperature,
            force=True,
        )
        submissions.append({"job_id": jid, "output_path": j["output_path"]})
    results: list[Path] = []
    for sub in submissions:
        client.poll(sub["job_id"], interval=0.5, timeout=1200)
        out_path: Path = sub["output_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(client.get_result_bytes(sub["job_id"]))
        if out_path.stat().st_size < 100:
            raise RuntimeError(f"tts-clone produced empty output for {out_path}")
        results.append(out_path)
    return results


__all__ = ["tts_design_to_file", "tts_clone_to_file", "tts_clone_many"]
