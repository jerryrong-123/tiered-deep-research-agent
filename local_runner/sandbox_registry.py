"""Durable per-trial sandbox ids, independent of the observational event log."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_LOCK = threading.RLock()


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _issue(raw, **location):
    return {**location, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class SandboxRegistry:
    """Write one atomically replaced, fsynced record per returned sandbox id.

    Register before exposing a newly created sandbox to its caller. A worker
    and its supervisor use the same trial directory; the supervisor reads after
    worker termination. This is not a general multi-process registry service.
    """

    def __init__(self, trial_directory):
        self.directory = Path(trial_directory) / "sandbox_registry"

    def _write(self, ident, status):
        if not isinstance(ident, str) or not ident:
            raise ValueError("sandbox_id_required")
        digest = hashlib.sha256(ident.encode()).hexdigest()
        with _LOCK:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / (digest + ".json")
            temporary = self.directory / (digest + "." + uuid4().hex + ".pending")
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump({"version": "sandbox_registry_v1", "sandbox_id": ident,
                               "status": status, "utc": _stamp()}, stream)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

    def register(self, ident):
        self._write(ident, "active")

    def confirm_cleanup(self, ident):
        self._write(ident, "cleanup_confirmed")

    def records(self):
        records, invalid = {}, []
        with _LOCK:
            for path in sorted(self.directory.glob("*.json")):
                raw = path.read_bytes()
                try:
                    row = json.loads(raw)
                    ident = row["sandbox_id"]
                    if (not isinstance(ident, str) or not ident
                            or row["status"] not in {"active", "cleanup_confirmed"}
                            or path.stem != hashlib.sha256(ident.encode()).hexdigest()):
                        raise ValueError("invalid_registry_record")
                except (ValueError, UnicodeError, KeyError, TypeError):
                    invalid.append(_issue(raw, file=path.name))
                    continue
                records[ident] = row
        return records, invalid


def read_sandbox_events(path):
    """Fallback for older event-only records; retain bad lines without trusting them."""
    pending, invalid = set(), []
    path = Path(path)
    if not path.exists():
        return pending, invalid
    for number, raw in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("event_not_object")
            kind = row.get("kind")
            if kind not in {"sandbox_created", "sandbox_cleanup"}:
                continue
            ident = row["sandbox_id"]
            if not isinstance(ident, str) or not ident:
                raise ValueError("sandbox_id_required")
        except (ValueError, UnicodeError, KeyError, TypeError):
            invalid.append(_issue(raw, line=number))
            continue
        if kind == "sandbox_created":
            pending.add(ident)
        elif row.get("confirmed") is True:
            pending.discard(ident)
    return pending, invalid
