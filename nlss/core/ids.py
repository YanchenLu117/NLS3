"""Stable, geometry-independent object and edge identifiers (V6).

The same canonical scientific object must always map to the same ID, so that
paraphrases that canonicalize to the same executable object collapse to the
same node.  IDs are derived only from the canonical form, never from a learned
embedding or geometry.
"""

from __future__ import annotations

import hashlib


def stable_object_id(task_id: str, canonical_form: str) -> str:
    """Derive a deterministic node ID from the task and canonical form."""
    raw = f"{task_id}\0{canonical_form}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:24]
    return f"obj_{digest}"


def stable_edge_id(source_id: str, target_id: str, relation_type: str) -> str:
    """Derive a deterministic edge ID from endpoints and relation type."""
    raw = f"{source_id}\0{target_id}\0{relation_type}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:24]
    return f"edge_{digest}"
