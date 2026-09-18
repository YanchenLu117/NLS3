"""invariants.py — substrate invariant audit for Explore/Evolve (Methods §4).

The activated substrate S=(H_S,F_S,Γ_S,Interp_S) is FROZEN across an
exploration episode. A new element (new hypothesis, new carrier, new operator,
new Γ pair, new interp rule) is a CERTIFIED ADMISSION, not a new degree of
freedom: it must enter through Γ_cert admission (Main Theorem II), never by
mutating the frozen structure.

Violation counters (paper claim: all = 0 on formal runs):
  space_type_change   the variant space's type (continuous/discrete/graph/…)
                      changed mid-episode without a SubstrateChallenge
  new_sort_insertion  a semantic sort was added outside admission
  gamma_mutation      an existing Γ pair was rewritten (not appended)
  interp_mutation     the interp rule changed behavior on identical input
Every loop turn records substrate_hash; any divergence vs the frozen hash that
is not accompanied by a certified admission record = violation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

VIOLATION_KINDS = ["space_type_change", "new_sort_insertion",
                   "gamma_mutation", "interp_mutation"]


@dataclass
class InvariantLedger:
    episode_id: str
    frozen_hash: str
    turns: list = field(default_factory=list)
    violations: dict = field(default_factory=lambda: {k: 0 for k in VIOLATION_KINDS})
    admission_count: int = 0   # certified Γ_cert admissions (allowed hash change)

    def record_turn(self, turn: int, chi: dict, event: str = "explore",
                    admission: dict | None = None) -> dict:
        """Record one loop turn. If chi's hash differs from frozen:
        legal ONLY with a certified admission (Γ_cert)."""
        from ..schema.chi_schema import canonical_substrate_hash
        h = canonical_substrate_hash(chi)
        rec = {"turn": turn, "event": event, "substrate_hash": h}
        if h != self.frozen_hash:
            if admission and admission.get("certified"):
                self.admission_count += 1
                rec["via"] = "certified_admission"
                rec["admission_id"] = admission.get("admission_id", "")
            else:
                rec["via"] = "VIOLATION"
                kind = _classify(chi, self._frozen_chi_hint())
                self.violations[kind] += 1
        self.turns.append(rec)
        return rec

    def _frozen_chi_hint(self) -> dict:
        return getattr(self, "_chi_ref", {}) or {}

    def attach_reference(self, chi: dict) -> None:
        self._chi_ref = chi

    def summary(self) -> dict:
        return {"episode_id": self.episode_id, "frozen_hash": self.frozen_hash,
                "n_turns": len(self.turns),
                "certified_admissions": self.admission_count,
                "violations": dict(self.violations),
                "clean": all(v == 0 for v in self.violations.values())}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": self.summary(),
                                    "turns": self.turns},
                                   ensure_ascii=False, indent=1))


def _classify(chi: dict, frozen: dict) -> str:
    """Heuristic classification of the first violation kind (audit detail)."""
    try:
        fc = frozenset(s.get("name") for s in frozen.get("content", {}).get("sorts", []))
        cc = frozenset(s.get("name") for s in chi.get("content", {}).get("sorts", []))
        if cc != fc:
            return "new_sort_insertion"
        fh = json.dumps(frozen.get("realization", {}), sort_keys=True)
        ch = json.dumps(chi.get("realization", {}), sort_keys=True)
        if fh != ch:
            return "gamma_mutation"
        if json.dumps(frozen.get("interp", {}), sort_keys=True) != \
           json.dumps(chi.get("interp", {}), sort_keys=True):
            return "interp_mutation"
    except Exception:
        pass
    return "space_type_change"
