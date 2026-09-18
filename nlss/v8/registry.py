"""V8 P0 — versioned preregistration registry (NLSS Experiment Detail V8 §2.2).

P0 produces one versioned, machine-readable ``preregistration_registry.json``
published and hashed BEFORE any final outcome is accessible.  An unset
mandatory field blocks the affected scored cell (per-cell blocking — never a
silent global pass); a registry can never be completed by a choice made after
comparative results are visible: after :meth:`PreregRegistry.finalize` the
document is frozen and content-addressed (``registry_hash`` = SHA-256 over the
canonical bytes of the versioned content envelope).

The registry is ordinary auditable data + validators (Operational Realization
Principle) — no formal-semantics machinery at runtime.

Structure (Detail §2.2):
- global mandatory sections: repository, datasets, corpus, model, prompts,
  legality, solutions, adapters;
- per-endpoint entries (``endpoints[endpoint_id]``): exact metric function,
  curve scalarization, favorable direction, delta_min/delta_eq/delta_harm,
  failure score, CI method, effect-size formula, resampling scheme,
  multiplicity family;
- per-track entries (``tracks[track_id]``): task list, min/max sample count,
  seed list, budget table, checkpoints, batch-fill algorithm;
- per-baseline entries (``baselines[baseline_id]``): official identifier,
  commit/container hash, entry point, complete config, permitted adapter diff,
  reproduction tolerance, objective gate result;
- scored cells (``register_cell``) bind method-task cells to one endpoint, one
  track, and the baselines they are compared against.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1

_TYPE_NOTE = "bool is rejected for numeric/str fields (bool subclasses int)"

_GLOBAL_MANDATORY: tuple[tuple[str, tuple[type, ...]], ...] = (
    ("repository.commits", (dict,)),
    ("repository.submodule_commits", (dict,)),
    ("datasets.oracle_hashes", (dict,)),
    ("datasets.dataset_hashes", (dict,)),
    ("corpus.task_dossier_hashes", (dict,)),
    ("corpus.frozen_corpus_hashes", (dict,)),
    ("model.model_id", (str,)),
    ("model.tokenizer_id", (str,)),
    ("model.temperature", (int, float)),
    ("model.tool_config", (dict,)),
    ("prompts.prompt_hashes", (dict,)),
    ("prompts.persistent_state_schemas", (dict,)),
    ("legality.candidate_legality_rules", (dict,)),
    ("legality.replacement_rules", (dict,)),
    ("solutions.thresholds", (dict,)),
    ("solutions.outcome_blind_geometry", (dict,)),
    ("budgets.stopping_rules", (dict,)),
    ("budgets.retry_policy", (dict,)),
    ("budgets.timeout_policy", (dict,)),
    ("budgets.failure_rules", (dict,)),
    ("seeds.initial_evidence_ids", (dict,)),
    ("adapters.identity_reports", (dict,)),
    ("adapters.supported_cells", (dict,)),
)

_ENDPOINT_MANDATORY: tuple[tuple[str, tuple[type, ...]], ...] = (
    ("metric_function", (str,)),
    ("curve_scalarization", (str,)),
    ("favorable_direction", (str,)),
    ("delta_min", (int, float)),
    ("delta_eq", (int, float)),
    ("delta_harm", (int, float)),
    ("failure_score", (int, float, str)),
    ("ci_method", (str,)),
    ("effect_size_formula", (str,)),
    ("resampling_scheme", (str,)),
    ("multiplicity_family", (str,)),
)

_TRACK_MANDATORY: tuple[tuple[str, tuple[type, ...]], ...] = (
    ("task_list", (list,)),
    ("min_samples", (int,)),
    ("max_samples", (int,)),
    ("seed_list", (list,)),
    ("budget_table", (dict,)),
    ("checkpoints", (list,)),
    ("batch_fill_algorithm", (str,)),
)

_BASELINE_MANDATORY: tuple[tuple[str, tuple[type, ...]], ...] = (
    ("official_identifier", (str,)),
    ("commit_hash", (str,)),
    ("container_hash", (str,)),  # commit OR container identity; both recorded
    ("entry_point", (str,)),
    ("config", (dict,)),
    ("permitted_adapter_diff", (str,)),
    ("reproduction_tolerance", (str,)),
    ("gate_result", (str,)),
)


class FrozenRegistryError(RuntimeError):
    """Raised on any mutation attempt after :meth:`PreregRegistry.finalize`."""


class BlockedCellError(RuntimeError):
    """A scored cell is blocked by unset mandatory registry fields."""


class RegistryHashMismatch(ValueError):
    """Published document fails its own content-address verification."""


def _check_type(value: Any, types: tuple[type, ...]) -> bool:
    if isinstance(value, bool) and bool not in types:
        return False  # bool subclasses int; reject unless explicitly allowed
    if not isinstance(value, types):
        return False
    if isinstance(value, (dict, list, str)) and len(value) == 0:
        return False  # §2.2: an unset mandatory field blocks — empty containers block too
    return True


def _missing_dotted(data: Mapping[str, Any], mandatory: Iterable[tuple[str, tuple[type, ...]]]) -> list[str]:
    missing: list[str] = []
    for dotted, types in mandatory:
        cur: Any = data
        found = True
        for part in dotted.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                found = False
                break
            cur = cur[part]
        if not found or not _check_type(cur, types):
            missing.append(dotted)
    return missing


def _entry_missing(entry: Any, mandatory: tuple[tuple[str, tuple[type, ...]], ...]) -> list[str]:
    if not isinstance(entry, Mapping):
        return [name for name, _ in mandatory]
    return [name for name, types in mandatory if name not in entry or not _check_type(entry[name], types)]


def canonical_registry_bytes(document: Mapping[str, Any]) -> bytes:
    """Canonical bytes hashed for the content envelope (excluding publish metadata)."""
    envelope = {
        "schema_version": document["schema_version"],
        "title": document["title"],
        "created_at": document["created_at"],
        "content": document["content"],
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class PreregRegistry:
    """Mutable until :meth:`finalize`, frozen and content-addressed after."""

    def __init__(self, title: str) -> None:
        self._document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "title": title,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "content": {
                "repository": {},
                "datasets": {},
                "corpus": {},
                "model": {},
                "prompts": {},
                "legality": {},
                "solutions": {},
                "adapters": {},
                "endpoints": {},
                "tracks": {},
                "baselines": {},
                "cells": {},
            },
        }
        self._finalized = False

    # ------------------------------------------------------------------ draft
    def set(self, dotted: str, value: Any) -> "PreregRegistry":
        self._ensure_mutable()
        parts = dotted.split(".")
        cur = self._document["content"]
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = copy.deepcopy(value)
        return self

    def set_endpoint(self, endpoint_id: str, spec: Mapping[str, Any]) -> "PreregRegistry":
        """Record an endpoint spec.  Incomplete specs are ACCEPTED and reported
        by :meth:`blocked_cells` for every cell bound to them (Detail §2.2:
        unset mandatory field blocks the affected scored cell — it does not
        silently pass, and P0 publication is not refused)."""
        self._ensure_mutable()
        self._document["content"]["endpoints"][endpoint_id] = copy.deepcopy(dict(spec))
        return self

    def set_track(self, track_id: str, spec: Mapping[str, Any]) -> "PreregRegistry":
        self._ensure_mutable()
        self._document["content"]["tracks"][track_id] = copy.deepcopy(dict(spec))
        return self

    def set_baseline(self, baseline_id: str, spec: Mapping[str, Any]) -> "PreregRegistry":
        self._ensure_mutable()
        self._document["content"]["baselines"][baseline_id] = copy.deepcopy(dict(spec))
        return self

    def register_cell(
        self,
        cell_id: str,
        *,
        endpoint_id: str,
        track_id: str,
        baseline_ids: Iterable[str] = (),
        axis_ids: Iterable[str] = (),
    ) -> "PreregRegistry":
        """Bind a scored cell to its endpoint, track, baselines, and the Pareto
        utility/resource axes it is compared on (Detail §2.2: every Pareto axis
        carries the full endpoint-level record)."""
        self._ensure_mutable()
        self._document["content"]["cells"][cell_id] = {
            "endpoint_id": endpoint_id,
            "track_id": track_id,
            "baseline_ids": sorted(set(baseline_ids)),
            "axis_ids": sorted(set(axis_ids)),
        }
        return self

    def set_pareto_axis(self, axis_id: str, spec: Mapping[str, Any]) -> "PreregRegistry":
        """Register a Pareto utility/resource axis with the SAME mandatory
        endpoint-level record (metric function, scalarization, direction,
        deltas, failure score, CI method, effect size, resampling, family)."""
        self._ensure_mutable()
        self._document["content"]["pareto_axes"][axis_id] = copy.deepcopy(dict(spec))
        return self

    # ------------------------------------------------------------- validation
    def blocked_cells(self) -> dict[str, list[str]]:
        """Map cell_id -> blocking reasons (Detail §2.2: per-cell blocking)."""
        content = self._document["content"]
        global_missing = _missing_dotted(content, _GLOBAL_MANDATORY)
        blocks: dict[str, list[str]] = {}
        for cell_id, cell in content["cells"].items():
            reasons: list[str] = []
            if global_missing:
                reasons.extend(f"global:{p}" for p in global_missing)
            if cell["endpoint_id"] not in content["endpoints"]:
                reasons.append(f"endpoint:{cell['endpoint_id']}:UNREGISTERED")
            else:
                for name in _entry_missing(content["endpoints"][cell["endpoint_id"]], _ENDPOINT_MANDATORY):
                    reasons.append(f"endpoint:{cell['endpoint_id']}:{name}")
            for axis_id in cell.get("axis_ids", []):
                if axis_id not in content["pareto_axes"]:
                    reasons.append(f"pareto_axis:{axis_id}:UNREGISTERED")
                else:
                    for name in _entry_missing(content["pareto_axes"][axis_id], _ENDPOINT_MANDATORY):
                        reasons.append(f"pareto_axis:{axis_id}:{name}")
            if cell["track_id"] not in content["tracks"]:
                reasons.append(f"track:{cell['track_id']}:UNREGISTERED")
            else:
                for name in _entry_missing(content["tracks"][cell["track_id"]], _TRACK_MANDATORY):
                    reasons.append(f"track:{cell['track_id']}:{name}")
            for baseline_id in cell["baseline_ids"]:
                if baseline_id not in content["baselines"]:
                    reasons.append(f"baseline:{baseline_id}:UNREGISTERED")
                else:
                    for name in _entry_missing(content["baselines"][baseline_id], _BASELINE_MANDATORY):
                        reasons.append(f"baseline:{baseline_id}:{name}")
            if reasons:
                blocks[cell_id] = reasons
        return blocks

    def assert_scoreable(self, cell_id: str) -> None:
        """Raise :class:`BlockedCellError` unless the cell may run scored."""
        if not self._finalized:
            raise FrozenRegistryError("registry not finalized; publish P0 before scored runs")
        blocks = self.blocked_cells().get(cell_id)
        if blocks:
            raise BlockedCellError(f"cell '{cell_id}' blocked by: {'; '.join(blocks)}")

    # ---------------------------------------------------------------- publish
    def finalize(self) -> dict[str, Any]:
        """Freeze, hash, and publish the registry document (idempotent)."""
        if self._finalized:
            return copy.deepcopy(self._document)
        digest = hashlib.sha256(canonical_registry_bytes(self._document)).hexdigest()
        self._document["published_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._document["registry_hash"] = f"sha256:{digest}"
        self._finalized = True
        return copy.deepcopy(self._document)

    def to_json(self, indent: int | None = 2) -> str:
        if not self._finalized:
            raise FrozenRegistryError("finalize() before serialization")
        return json.dumps(self._document, sort_keys=True, indent=indent, ensure_ascii=False)

    def _ensure_mutable(self) -> None:
        if self._finalized:
            raise FrozenRegistryError("registry is finalized and frozen (Detail §2.2)")

    @classmethod
    def from_published(cls, document: Mapping[str, Any]) -> "PreregRegistry":
        """Wrap a published (hash-verified) registry document as a frozen instance."""
        verify_registry_hash(document)
        instance = cls(str(document.get("title", "published")))
        instance._document = copy.deepcopy(dict(document))
        instance._finalized = True
        return instance


def load_registry(path: str) -> dict[str, Any]:
    """Load a published registry document from JSON and verify its hash."""
    with open(path, "r", encoding="utf-8") as fh:
        document = json.load(fh)
    verify_registry_hash(document)
    return document


def verify_registry_hash(document: Mapping[str, Any]) -> None:
    """Recompute the content hash; raise on any post-publish tampering."""
    recorded = document.get("registry_hash")
    if not isinstance(recorded, str) or not recorded.startswith("sha256:"):
        raise RegistryHashMismatch("document carries no registry_hash")
    digest = hashlib.sha256(canonical_registry_bytes(document)).hexdigest()
    if digest != recorded.split(":", 1)[1]:
        raise RegistryHashMismatch(f"hash mismatch: recorded {recorded}, computed sha256:{digest}")
