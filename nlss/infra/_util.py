"""Small shared utilities for the infra package (stdlib only).

Everything here is deliberately dependency-free so the package runs
identically in every project environment (nlss-core, made-gpu, vcc-4
base python) and on both servers (c89 / vcc-4).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]


def atomic_write_json(path: PathLike, obj: Any) -> None:
    """Write ``obj`` as JSON atomically (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: PathLike, default: Any = None) -> Any:
    """Read JSON, returning ``default`` on any error (missing/corrupt)."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def append_jsonl(path: PathLike, obj: Any) -> None:
    """Append one JSON object as a line (creates parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")


def sha256_file(path: PathLike) -> str:
    """Hex sha256 of a file, streaming in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_sha_prefix(token: str) -> str:
    """``sha256:<hex>`` or ``<hex>`` -> ``<hex>``."""
    prefix = "sha256:"
    return token[len(prefix):] if token.startswith(prefix) else token


def read_api_key(env_name: Optional[str] = None, file_path: Optional[str] = None) -> str:
    """Resolve an API key from an env var or a file (first line, stripped)."""
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    if file_path:
        value = Path(file_path).read_text().strip()
        if value:
            return value
    raise SystemExit("api key: pass --api-key-env NAME or --api-key-file PATH")


def log_line(message: str, stream: Any = None) -> None:
    """Timestamped single-line log to stderr (unbuffered)."""
    import sys

    sink = stream or sys.stderr
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{stamp}] {message}", file=sink, flush=True)
