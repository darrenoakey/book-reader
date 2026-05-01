from pathlib import Path

from arbiter_client import ArbiterClient


# ##################################################################
# tts design to file
# generate a voice from description and save the wav locally
def tts_design_to_file(description: str, text: str, output_path: Path,
                       language: str = "English", temperature: float = 0.9) -> Path:
    client = ArbiterClient(timeout=60)
    job_id = client.submit(
        "tts-design",
        text=text,
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
# tts design many
# submit tts-design jobs in parallel and write results sequentially
def tts_design_many(jobs: list[dict], language: str = "English",
                    temperature: float = 0.9) -> list[Path]:
    client = ArbiterClient(timeout=60)
    submissions: list[dict] = []
    for j in jobs:
        jid = client.submit(
            "tts-design",
            text=j["text"],
            instruct=j["description"],
            language=language,
            temperature=temperature,
            force=True,
        )
        submissions.append({"job_id": jid, "output_path": j["output_path"]})
    results: list[Path] = []
    for sub in submissions:
        client.poll(sub["job_id"], interval=0.5, timeout=900)
        out_path: Path = sub["output_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(client.get_result_bytes(sub["job_id"]))
        if out_path.stat().st_size < 100:
            raise RuntimeError(f"tts-design produced empty output for {out_path}")
        results.append(out_path)
    return results


__all__ = ["tts_design_to_file", "tts_design_many"]
