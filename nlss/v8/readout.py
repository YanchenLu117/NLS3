"""V8 P0-c — solution-space readout schema (Detail §3).

The external readout uses ONE frozen schema with 11 fields.  Fields may be
empty but cannot be omitted.  Every claim links to evidence or explicitly
abstains.  The runtime treats this as ordinary data + a validator.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

READOUT_SCHEMA_VERSION = 1

READOUT_FIELDS: tuple[str, ...] = (
    "observed_supported",
    "inferred_untested",
    "equivalence_groups",
    "solution_regions",
    "boundaries",
    "coverage_gaps",
    "uncertainties",
    "semantic_gaps",
    "realization_gaps",
    "provenance",
    "abstentions",
)

# Fields that carry claims (must be evidence-linked via provenance or abstained)
_CLAIM_FIELDS: frozenset[str] = frozenset(READOUT_FIELDS) - {"provenance", "abstentions"}


class ReadoutValidationError(ValueError):
    """Readout payload violates the frozen schema (Detail §3)."""


def build_readout(**fields: Any) -> dict[str, Any]:
    """Construct a readout dict; unknown keys are rejected, missing keys default to [].

    Entry convention: claim entries are mappings carrying a unique ``id``;
    ``provenance`` entries are mappings ``{"id", "claim_ids", "evidence_ids"}``;
    ``abstentions`` entries are mappings ``{"id", "claim_ids", "reason"}``.
    """
    unknown = sorted(set(fields) - set(READOUT_FIELDS))
    if unknown:
        raise ReadoutValidationError(f"unknown readout fields: {', '.join(unknown)}")
    readout: dict[str, Any] = {"schema_version": READOUT_SCHEMA_VERSION}
    for name in READOUT_FIELDS:
        value = copy.deepcopy(fields.get(name, []))
        readout[name] = value if isinstance(value, list) else [value]
    return readout


def validate_readout(payload: Mapping[str, Any]) -> list[str]:
    """Return a list of schema problems; an empty list means the payload is valid.

    Enforced invariants (Detail §3):
    1. all 11 fields present (empty allowed, omission is an error);
    2. every field is a list;
    3. claim entries are mappings with a unique non-empty ``id``;
    4. every claim is either evidence-linked (referenced by a provenance entry
       with non-empty ``evidence_ids``) or explicitly abstained;
    5. provenance entries carry non-empty ``evidence_ids``; abstention entries
       carry a non-empty ``reason``.
    """
    problems: list[str] = []
    for name in READOUT_FIELDS:
        if name not in payload:
            problems.append(f"missing field: {name} (empty allowed, omission forbidden)")
    if problems:
        return problems

    claim_ids: dict[str, str] = {}
    for name in sorted(_CLAIM_FIELDS):
        entries = payload[name]
        if not isinstance(entries, list):
            problems.append(f"field {name} must be a list")
            continue
        for entry in entries:
            if not isinstance(entry, Mapping) or not entry.get("id"):
                problems.append(f"field {name}: entries must be mappings with non-empty 'id'")
                continue
            entry_id = str(entry["id"])
            if entry_id in claim_ids:
                problems.append(f"duplicate claim id '{entry_id}' ({claim_ids[entry_id]} vs {name})")
            else:
                claim_ids[entry_id] = name

    provenance = payload["provenance"]
    abstentions = payload["abstentions"]
    if not isinstance(provenance, list) or not isinstance(abstentions, list):
        problems.append("fields provenance/abstentions must be lists")
        return problems

    linked: set[str] = set()
    for entry in provenance:
        if not isinstance(entry, Mapping):
            problems.append("provenance: entries must be mappings")
            continue
        evidence_ids = entry.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            problems.append(f"provenance '{entry.get('id', '?')}': evidence_ids must be a non-empty list")
        claim_ids_ref = entry.get("claim_ids", [])
        if isinstance(claim_ids_ref, list):
            linked.update(str(c) for c in claim_ids_ref)
    for entry in abstentions:
        if not isinstance(entry, Mapping):
            problems.append("abstentions: entries must be mappings")
            continue
        if not entry.get("reason"):
            problems.append(f"abstention '{entry.get('id', '?')}': reason must be non-empty")
        claim_ids_ref = entry.get("claim_ids", [])
        if isinstance(claim_ids_ref, list):
            linked.update(str(c) for c in claim_ids_ref)

    for claim_id, field in sorted(claim_ids.items()):
        if claim_id not in linked:
            problems.append(
                f"claim '{claim_id}' ({field}): no evidence link and no explicit abstention (Detail §3)"
            )
    return problems


def readout_to_json(payload: Mapping[str, Any], *, indent: int | None = 2) -> str:
    problems = validate_readout(payload)
    if problems:
        raise ReadoutValidationError("; ".join(problems))
    import json

    return json.dumps(payload, sort_keys=True, indent=indent, ensure_ascii=False)
