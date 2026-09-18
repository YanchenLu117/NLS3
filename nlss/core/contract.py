"""V7 Core — versioned contract diffing (infrastructure, EXPERIMENT_PLAN §2.1/§12.2).

The plan requires a versioned, machine-parseable representation contract and a
``schema_diff_v{n-1}_v{n}.json`` artifact per revision.  ``contract_diff`` is a
pure, stateless helper that reports added/removed/changed components between two
versions of the contract (paper P_t = (Sigma, R, T, Phi, G, K)).

The ``version`` is supplied by the caller (e.g. the archiver's contract-file
count), NOT parsed from ``spec_id`` — a revised spec's id (``s0.r``) contains no
numeric revision, so including it here would be both wrong and crash-prone.
"""

from __future__ import annotations

from typing import Any, Sequence

from .representation import RepresentationSpecification


def _diff_names(old_names: Sequence[str], new_names: Sequence[str]) -> dict[str, list[str]]:
    old_set, new_set = set(old_names), set(new_names)
    return {
        "added": sorted(new_set - old_set),
        "removed": sorted(old_set - new_set),
    }


def _diff_dict(
    old: dict | None,
    new: dict | None,
) -> dict[str, Any]:
    """Key-level diff of geometry/grounding maps (added/removed/changed keys)."""
    old = {str(k): v for k, v in (old or {}).items()}
    new = {str(k): v for k, v in (new or {}).items()}
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
    }


def contract_diff(
    old: RepresentationSpecification | None,
    new: RepresentationSpecification,
    *,
    version: int = 1,
) -> dict[str, Any]:
    """Return a JSON-serializable diff between an old and new contract.

    ``old`` may be None for the very first version (round 0).  ``version`` is
    the version of ``new`` and is supplied by the caller (the archiver), so the
    reported number is the real revision, never derived from ``spec_id``.
    """
    if old is None:
        old_vars = old_rels = old_trs = old_ents = ()
        old_geom = old_ground = None
        old_hint = None
    else:
        old_vars = tuple(v.name for v in old.typed_variables)
        old_rels = tuple(r.name for r in old.relations)
        old_trs = tuple(t.name for t in old.transforms)
        old_ents = old.entities
        old_geom = old.geometry
        old_ground = old.grounding
        old_hint = old.backend_hint

    return {
        "version": int(version),
        "typed_variables": _diff_names(old_vars, tuple(v.name for v in new.typed_variables)),
        "relations": _diff_names(old_rels, tuple(r.name for r in new.relations)),
        "transforms": _diff_names(old_trs, tuple(t.name for t in new.transforms)),
        "entities": _diff_names(old_ents, new.entities),
        "geometry": _diff_dict(old_geom, new.geometry),
        "grounding": _diff_dict(old_ground, new.grounding),
        "backend_changed": old_hint != new.backend_hint,
    }
