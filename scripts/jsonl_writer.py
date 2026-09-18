"""Crash-safe JSONL append writer (B5 fix, 2026-08).

Root cause of the 8/8 malformed lines in work/b5/results.jsonl: the writer
was a shell echo of a hand-built printf template that spliced the raw
V3_BEST R2=... rl=[...] expr=... params=[...] log tail straight after
"r2": (no quotes, no commas), emitted "r2": with an empty value when
log extraction failed, and used unlocked >> appends that could interleave
or half-flush under concurrency.

This module replaces that path: one json.dumps record per line, flushed
and fsynced on every append, serialized across processes with an exclusive
fcntl.flock.  allow_nan=False keeps NaN/Infinity (invalid JSON) out of
the file -- callers must sanitize or omit such fields.
"""
from __future__ import annotations

import fcntl
import json
import os

__all__ = ["append_jsonl", "JsonlWriter"]


class JsonlWriter:
    """Append-only JSONL writer: line-atomic under concurrency, durable per write."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def append(self, record: dict) -> str:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with open(self.path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return line


def append_jsonl(path: str, record: dict) -> str:
    """Append one record as a single JSON line; returns the written line."""
    return JsonlWriter(path).append(record)
