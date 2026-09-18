"""V8 — R^run_t carrier (methods §3.4-3.9, Detail §3).

At round t NLSS materializes

    R^run_t = (G_t, R_t, C_t, P_t, g⁰_t, Γ_t)

plus Q^F_t (outcome-blind fidelity probes) and the append-only ledger D_t
(carried by :class:`nlss.v8.artifacts.EvidenceLedger`).  This module is the
typed in-memory carrier:

- G_t: generators with type/description/scope/provenance;
- R_t: certified relations (equality/refinement/conjunction/distinction);
- C_t: proposed exhaustive covers with (optional) certificates;
- P_t: executable representation contract id + version;
- g⁰_t: generator→computational-region grounding rules;
- Γ_t: materialized hypothesis/state correspondences — PARTIAL and
  MULTIVALUED by design (one h may ground to several z; entries carry
  evidence links); uncovered entries are reported, never dropped.

Gap classification (Detail §3.2 / methods §3.9): SEMANTIC_GAP when the
hypothesis is not representable under the current generators (⟦h⟧ ∉ O^rep);
REALIZATION_GAP when it is represented but g⁻¹(h) = ⊥_C (no legal executable
state).  Ordinary data + validators — no category machinery at runtime.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

GAP_SEMANTIC = "SEMANTIC_GAP"
GAP_REALIZATION = "REALIZATION_GAP"
GAP_NONE = "NONE"


@dataclass(frozen=True)
class Generator:
    gen_id: str
    type: str
    description: str
    scope: str
    provenance_ref: str
    predicate_ref: str  # public deterministic DSL predicate (Detail §4.1)


@dataclass(frozen=True)
class DeclaredRelation:
    relation_id: str
    kind: str  # equality | refinement | conjunction | distinction
    operands: tuple[str, ...]  # generator ids
    certificate: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Cover:
    cover_id: str
    parent_ref: str
    member_refs: tuple[str, ...]
    certificate: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GroundingRule:
    """g⁰_t entry: generator → computational region (rule reference)."""

    gen_id: str
    region_id: str
    rule_ref: str


@dataclass
class GammaEntry:
    """One materialized (h, z) correspondence — partial/multivalued allowed."""

    h_id: str
    z_id: str
    evidence_ids: tuple[str, ...] = ()
    covered: bool | None = None  # None = not yet checked against certified regions

    def to_dict(self) -> dict[str, Any]:
        return {
            "h_id": self.h_id,
            "z_id": self.z_id,
            "evidence_ids": list(self.evidence_ids),
            "covered": self.covered,
        }


class RepresentationState:
    """Mutable within a round; snapshot-hashable for archiving."""

    def __init__(
        self,
        *,
        contract_id: str,
        contract_version: int,
        generators: tuple[Generator, ...] = (),
        relations: tuple[DeclaredRelation, ...] = (),
        covers: tuple[Cover, ...] = (),
        grounding_rules: tuple[GroundingRule, ...] = (),
        gamma: tuple[GammaEntry, ...] = (),
        fidelity_probes_ref: str | None = None,
    ) -> None:
        self.contract_id = contract_id
        self.contract_version = contract_version
        self.generators = tuple(generators)
        self.relations = tuple(relations)
        self.covers = tuple(covers)
        self.grounding_rules = tuple(grounding_rules)
        self.gamma = list(gamma)
        self.fidelity_probes_ref = fidelity_probes_ref
        self._index()

    def _index(self) -> None:
        self._gen_ids = {g.gen_id for g in self.generators}

    # ------------------------------------------------------------- Γ_t append
    def materialize_correspondence(self, h_id: str, z_id: str, evidence_ids: tuple[str, ...] = ()) -> GammaEntry:
        """Append one (h, z) correspondence to Γ_t (append-only within the
        round).  Coverage is evaluated against certified regions by
        :meth:`uncovered` — not at append time."""
        entry = GammaEntry(h_id=h_id, z_id=z_id, evidence_ids=evidence_ids, covered=None)
        self.gamma.append(entry)
        return entry

    def uncovered(self, certified_regions: set[str]) -> list[GammaEntry]:
        """Γ entries whose z does not lie in a certified computational region
        (Detail §4.2 condition 5: unrepresented pairs cannot support Level 2)."""
        for entry in self.gamma:
            entry.covered = entry.z_id in certified_regions
        return [entry for entry in self.gamma if not entry.covered]

    def classify_gap(self, h_id: str, *, representable: bool, realizable: bool) -> str:
        """GAP routing (Detail §3.2 GROUNDING subclassification).

        SEMANTIC_GAP: h not representable under the current presentation;
        REALIZATION_GAP: represented but g⁻¹(h) = ⊥_C (no legal executable
        state); NONE when grounded and realizable."""
        if not representable:
            return GAP_SEMANTIC
        if not realizable:
            return GAP_REALIZATION
        return GAP_NONE

    # ------------------------------------------------------------------ misc
    def snapshot_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return f"sha256:{hashlib.sha256(blob.encode('utf-8')).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "generators": [
                {
                    "gen_id": g.gen_id,
                    "type": g.type,
                    "description": g.description,
                    "scope": g.scope,
                    "provenance_ref": g.provenance_ref,
                    "predicate_ref": g.predicate_ref,
                }
                for g in self.generators
            ],
            "relations": [
                {
                    "relation_id": r.relation_id,
                    "kind": r.kind,
                    "operands": list(r.operands),
                    "certificate": r.certificate,
                }
                for r in self.relations
            ],
            "covers": [
                {"cover_id": c.cover_id, "parent_ref": c.parent_ref, "member_refs": list(c.member_refs), "certificate": c.certificate}
                for c in self.covers
            ],
            "grounding_rules": [
                {"gen_id": r.gen_id, "region_id": r.region_id, "rule_ref": r.rule_ref} for r in self.grounding_rules
            ],
            "gamma": [
                {"h_id": e.h_id, "z_id": e.z_id, "evidence_ids": list(e.evidence_ids), "covered": e.covered}
                for e in self.gamma
            ],
            "fidelity_probes_ref": self.fidelity_probes_ref,
        }
