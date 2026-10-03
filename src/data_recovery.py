"""Authoritative resilient data contract.

Two disjoint failure classes exist and are never mixed:

* **Data issues** (content or model output is unknown, malformed, ambiguous, overlarge,
  empty, oddly encoded, cyclic ...). They are *recoverable*: the offending item or chapter
  is quarantined with exact evidence and a checkpoint into ``data_recovery.jsonl``, the
  run continues with the next item, and nothing is silently dropped, invented, or turned
  into a global alias.
* **Operational failures** (store/infrastructure integrity: unreadable or corrupt project
  state, I/O errors, exhausted providers). They fail closed as a typed
  :class:`OperationalError`; there is no fallback store and no broad swallow.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

LEDGER_NAME = "data_recovery.jsonl"
REVALIDATION_ATTEMPT_LIMIT = 3
EVIDENCE_LIMIT = 600
EVIDENCE_MAX_DEPTH = 6
EVIDENCE_MAX_ITEMS = 50
RAW_BYTES_LIMIT = 4096


class DataIssue(RuntimeError):
    """A recoverable content/model data problem with a stable machine-readable code."""

    def __init__(self, code: str, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.code = code
        self.evidence = evidence or {}


class OperationalError(RuntimeError):
    """Infrastructure or store integrity failure. Always fail closed; never recovered per item."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


# Builtin exception types that only ever describe malformed *data* when raised while
# processing one item. OSError (apart from decode errors) is deliberately absent.
DATA_ERRORS: tuple[type[BaseException], ...] = (
    DataIssue,
    ValueError,  # includes json.JSONDecodeError and UnicodeError
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    RecursionError,
    OverflowError,
)


def is_data_error(error: BaseException) -> bool:
    return isinstance(error, DATA_ERRORS)


def scope_digest(value: object) -> str:
    """Exact, total digest of a scope/hash value, matching the ledger's de-duplication keys."""
    return _digest(_canonical(value))


def bounded(value: object, limit: int = EVIDENCE_LIMIT) -> str:
    """Printable, JSON-safe, length-bounded rendering (lone surrogates and control chars escaped)."""
    text = value if isinstance(value, str) else repr(value)
    text = text.encode("unicode_escape", errors="backslashreplace").decode("ascii")
    return text if len(text) <= limit else text[:limit] + f"...[{len(text)} chars]"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object, seen: tuple[int, ...] = ()) -> bytes:
    """Total, unbounded, deterministic byte rendering of any value (cycles marked, surrogates kept)."""
    if isinstance(value, bytes | bytearray | memoryview):
        return b"b:" + bytes(value)
    if isinstance(value, str):
        return b"s:" + value.encode("utf-8", errors="surrogatepass")
    if value is None or isinstance(value, bool | int):
        return b"p:" + repr(value).encode("ascii")
    if isinstance(value, float):
        return b"f:" + repr(value).encode("ascii")
    if id(value) in seen:
        return b"<cycle>"
    inner = (*seen, id(value))
    try:
        if isinstance(value, dict):
            parts = sorted(_canonical(k, inner) + b"=" + _canonical(v, inner) for k, v in value.items())
            return b"{" + b",".join(parts) + b"}"
        if isinstance(value, list | tuple):
            return b"[" + b",".join(_canonical(v, inner) for v in value) + b"]"
        if isinstance(value, set | frozenset):
            return b"<" + b",".join(sorted(_canonical(v, inner) for v in value)) + b">"
    except RecursionError:
        return b"<too-deep>"
    try:
        return b"o:" + type(value).__qualname__.encode() + b":" + repr(value).encode("utf-8", errors="surrogatepass")
    except Exception:  # noqa: BLE001 - arbitrary __repr__ of an unsupported shape must not break recording
        return b"o:" + type(value).__qualname__.encode() + b":<unrepresentable>"


def payload_hash(value: object) -> str:
    """Full-payload hash over the unbounded canonical form (independent of any bounding)."""
    return _digest(_canonical(value))


def safe_evidence(value: object, depth: int = 0, seen: tuple[int, ...] = ()) -> object:
    """JSON-safe, bounded, recursive rendering. Total: never raises for any input shape."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__float__": repr(value)}
    if isinstance(value, str):
        text = bounded(value)
        try:
            value.encode("utf-8")
            clean = len(value) <= EVIDENCE_LIMIT
        except UnicodeEncodeError:
            clean = False
        if not clean:
            return {"text": text, "sha256": _digest(_canonical(value)), "chars": len(value)}
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raw = bytes(value)
        out: dict = {"__bytes__": len(raw), "sha256": _digest(raw)}
        out["base64"] = base64.b64encode(raw[:RAW_BYTES_LIMIT]).decode("ascii")
        if len(raw) > RAW_BYTES_LIMIT:
            out["truncated"] = True
        return out
    if id(value) in seen:
        return {"__cycle__": type(value).__name__}
    if depth >= EVIDENCE_MAX_DEPTH:
        return {"__depth_limit__": type(value).__name__, "sha256": payload_hash(value)}
    inner = (*seen, id(value))
    if isinstance(value, dict):
        items = list(value.items())
        out = {}
        for key, item in items[:EVIDENCE_MAX_ITEMS]:
            name = key if isinstance(key, str) and bounded(key, 100) == key else bounded(key, 100)
            while name in out:
                name += "'"
            out[name] = safe_evidence(item, depth + 1, inner)
        if len(items) > EVIDENCE_MAX_ITEMS:
            out["__truncated_items__"] = len(items) - EVIDENCE_MAX_ITEMS
        return out
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        shown = [safe_evidence(v, depth + 1, inner) for v in items[:EVIDENCE_MAX_ITEMS]]
        if len(items) > EVIDENCE_MAX_ITEMS:
            shown.append({"__truncated_items__": len(items) - EVIDENCE_MAX_ITEMS})
        return shown
    try:
        text = bounded(value)
    except Exception:  # noqa: BLE001 - arbitrary __repr__
        text = "<unrepresentable>"
    return {"__type__": type(value).__qualname__, "repr": text, "sha256": payload_hash(value)}


def error_evidence(error: BaseException) -> dict:
    evidence = {"error_type": type(error).__name__, "message": bounded(str(error))}
    if isinstance(error, DataIssue):
        evidence.update(error.evidence)
    return evidence


def error_code(error: BaseException) -> str:
    return error.code if isinstance(error, DataIssue) else type(error).__name__


def row_key(row: dict) -> tuple[str, str, str]:
    """Identity of one recorded row independent of its position: item, code and exact source hash digest."""
    return (row["item_sha256"], row["code"], row.get("dedup_hash", ""))


def row_scope(row: dict) -> tuple[object, object]:
    """The exact source names and hashes a row recorded (current ``source``/``source_hash``, else legacy ``chapters``/``raw_chapter_sha256``)."""
    evidence = row["evidence"] if isinstance(row["evidence"], dict) else {}
    return (
        evidence.get("source") or evidence.get("chapters"),
        evidence.get("source_hash") or evidence.get("raw_chapter_sha256"),
    )


def _attempted_hashes(attempt_row: dict) -> object:
    """The exact source hashes an attempt row was made against (its counter excluded)."""
    recorded = attempt_row["evidence"].get("source_hash")
    return recorded.get("exact") if isinstance(recorded, dict) else None


def _row_key(row: dict) -> tuple[str, str, str, str, str]:
    return (
        row["stage"],
        row.get("item_sha256", row["item"]),
        row["code"],
        row.get("dedup_scope", ""),
        row.get("dedup_hash", ""),
    )


@dataclass
class RecoveryLedger:
    """Durable, append-only, de-duplicated record of every recovered data issue for one project."""

    project: Path
    _seen: set[tuple[str, str, str, str, str]] = field(default_factory=set, init=False)
    # keys re-presented to any ledger instance (deduplicated); lets a replay see its issue recur
    presented: ClassVar[set[tuple[str, str, str, str, str]]] = set()

    def __post_init__(self) -> None:
        self.project = Path(self.project)
        self._seen = {_row_key(r) for r in self.entries()}

    @property
    def path(self) -> Path:
        return self.project / LEDGER_NAME

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as error:
            raise OperationalError(
                "recovery_ledger_unreadable", f"recovery ledger is unreadable: {self.path}"
            ) from error
        if not all(isinstance(r, dict) and {"stage", "item", "code"} <= set(r) for r in rows):
            raise OperationalError("recovery_ledger_corrupt", f"recovery ledger has malformed records: {self.path}")
        return rows

    def record(
        self,
        stage: str,
        item: str,
        code: str,
        message: str,
        *,
        severity: str = "warning",
        evidence: dict | None = None,
        checkpoint: dict | None = None,
    ) -> dict:
        """Persist one recovered issue. severity: warning | quarantine.

        Total over evidence/checkpoint shape: both are rendered JSON-safe and bounded while the full
        payload hash is kept. Idempotent per stage/item/code plus exact source scope and hash.
        """
        evidence = {} if evidence is None else evidence
        checkpoint = {} if checkpoint is None else checkpoint
        item_text = item if isinstance(item, str) else bounded(item)
        scope = evidence.get("source", evidence.get("scope")) if isinstance(evidence, dict) else None
        source_hash = None
        if isinstance(evidence, dict):
            source_hash = next((evidence[k] for k in ("source_hash", "sha256", "hash") if k in evidence), None)
        scope_text = "" if scope is None else _digest(_canonical(scope))
        hash_text = "" if source_hash is None else _digest(_canonical(source_hash))
        row = {
            "stage": bounded(stage, 100),
            "item": bounded(item, 200),
            "item_sha256": _digest(_canonical(item_text)),
            "code": bounded(code, 100),
            "severity": bounded(severity, 50),
            "message": bounded(message),
            "message_sha256": _digest(_canonical(message)),
            "evidence": safe_evidence(evidence),
            "evidence_sha256": payload_hash(evidence),
            "checkpoint": safe_evidence(checkpoint),
            "checkpoint_sha256": payload_hash(checkpoint),
            "dedup_scope": scope_text,
            "dedup_hash": hash_text,
        }
        key = _row_key(row)
        RecoveryLedger.presented.add(key)
        if key in self._seen:
            return row
        try:
            self.project.mkdir(parents=True, exist_ok=True)
            line = json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n"
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise OperationalError(
                "recovery_ledger_unwritable", f"cannot persist recovery ledger: {self.path}"
            ) from error
        self._seen.add(key)
        return row

    def open_rows(self, stage: str, severities: tuple[str, ...] = ("pending", "quarantine")) -> list[dict]:
        """Unresolved rows of a stage and severity with no durable resolution row for the same item/code/source hash."""
        rows = self.entries()
        resolved = {
            (
                r["stage"],
                r["item_sha256"],
                r["evidence"].get("resolves_code"),
                r["evidence"].get("resolves_dedup_hash", r.get("dedup_hash", "")),
            )
            for r in rows
            if r["severity"] == "resolved"
        }
        return [
            r
            for r in rows
            if r["stage"] == stage
            and r["severity"] in severities
            and (r["stage"], r["item_sha256"], r["code"], r.get("dedup_hash", "")) not in resolved
        ]

    def open_pending(self, stage: str) -> list[dict]:
        """Pending rows of a stage with no durable resolution row for the same item/code/source hash."""
        return self.open_rows(stage, ("pending",))

    def open_quarantined(self, stage: str) -> list[dict]:
        """Quarantine rows of a stage with no durable resolution row for the same item/code/source hash."""
        return self.open_rows(stage, ("quarantine",))

    def resolve(self, row: dict, outcome: str, detail: dict | None = None) -> dict:
        """Append (never rewrite) the durable resolution of one pending/quarantine row; history stays intact."""
        names, hashes = row_scope(row)
        return self.record(
            row["stage"],
            row["item"],
            # The resolved code is part of the row identity: two codes pending on one item and source must each
            # be closable, otherwise the second resolution is dropped as a duplicate of the first.
            f"{row['severity']}_resolved:{row['code']}",
            f"{row['severity']} {row['code']} resolved: {outcome}",
            severity="resolved",
            evidence={
                "source": names,
                "source_hash": hashes,
                "resolves_code": row["code"],
                "resolves_dedup_hash": row.get("dedup_hash", ""),
                "outcome": outcome,
                **(detail or {}),
            },
        )

    def attempts(self, row: dict, rows: list[dict] | None = None) -> list[dict]:
        """Durable replay-attempt rows for exactly this row's item, code and recorded source hashes."""
        hashes = row_scope(row)[1]
        return [
            r
            for r in (self.entries() if rows is None else rows)
            if r["stage"] == row["stage"]
            and r["severity"] == "replay_attempt"
            and r["item_sha256"] == row["item_sha256"]
            and r["evidence"].get("replays") == row["code"]
            and _attempted_hashes(r) == hashes
        ]

    def record_attempt(self, row: dict, detail: dict | None = None) -> dict:
        """Append the next bounded replay-attempt row for this row before any work is done (counts survive crashes)."""
        names, hashes = row_scope(row)
        number = len(self.attempts(row)) + 1
        return self.record(
            row["stage"],
            row["item"],
            f"{row['severity']}_replay_attempt:{row['code']}",  # code-qualified so sibling rows never dedupe each other
            f"replay attempt {number}",
            severity="replay_attempt",
            evidence={
                "replays": row["code"],
                "source": names,
                "source_hash": {"attempt": number, "exact": hashes},
                **(detail or {}),
            },
        )

    def revalidate(
        self,
        stage: str,
        exact: Callable[[dict], bool],
        retry: Callable[[dict], tuple[str, dict] | None],
        *,
        severities: tuple[str, ...] = ("quarantine",),
        attempt_limit: int = REVALIDATION_ATTEMPT_LIMIT,
        limit: int | None = None,
        defer: Callable[[dict], bool] | None = None,
    ) -> dict:
        """Bounded exact-hash retry of every unresolved row of a stage; history is only ever appended to.

        A row is retried only when ``exact(row)`` proves its recorded source still has its recorded hash, at most
        ``attempt_limit`` times per row and ``limit`` attempts per call. ``retry(row)`` returns ``(outcome, detail)``
        when the row is now validly handled (a durable resolution row is appended) or None when it still fails (the
        row stays open with its exact evidence). Stale, exhausted and deferred rows (``defer(row)``, e.g. just recorded this
        run) stay open and are counted.
        """
        summary = {"resolved": 0, "open": 0, "stale": 0, "exhausted": 0, "deferred": 0, "attempts": 0}
        snapshot = self.entries()
        for row in self.open_rows(stage, severities):
            if not exact(row):
                summary["stale"] += 1
                summary["open"] += 1
            elif len(self.attempts(row, snapshot)) >= attempt_limit:
                summary["exhausted"] += 1
                summary["open"] += 1
            elif (limit is not None and summary["attempts"] >= limit) or (defer is not None and defer(row)):
                summary["deferred"] += 1
                summary["open"] += 1
            else:
                snapshot.append(self.record_attempt(row))
                summary["attempts"] += 1
                result = retry(row)
                if result is None:
                    summary["open"] += 1
                else:
                    self.resolve(row, result[0], result[1])
                    summary["resolved"] += 1
        return summary

    def record_error(
        self,
        stage: str,
        item: str,
        error: BaseException,
        *,
        severity: str = "quarantine",
        evidence: dict | None = None,
        checkpoint: dict | None = None,
    ) -> dict:
        return self.record(
            stage,
            item,
            error_code(error),
            str(error),
            severity=severity,
            evidence={**error_evidence(error), **(evidence or {})},
            checkpoint=checkpoint,
        )


def operational(error: BaseException, status: str, what: str) -> OperationalError:
    """Wrap an infrastructure/store failure in the typed fail-closed status."""
    return OperationalError(status, f"{what}: {type(error).__name__}: {bounded(str(error), 200)}")


def load_json_store(path: Path, label: str, *, default: object = None, expect: type = dict) -> object:
    """Read an authoritative project store file; unreadable/corrupt/mistyped state fails closed (typed)."""
    path = Path(path)
    if not path.exists():
        if default is None:
            raise OperationalError(f"{label}_missing", f"required {label} is missing: {path}")
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OperationalError(f"{label}_unreadable", f"{label} is unreadable or corrupt: {path}") from error
    if not isinstance(value, expect):
        raise OperationalError(f"{label}_invalid", f"{label} is not a {expect.__name__}: {path}")
    return value
