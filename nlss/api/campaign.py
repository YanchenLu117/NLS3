"""V7 shared scientific-campaign API (EXPERIMENT_BENCHMARKS_BASELINES_FINAL §4).

The versioned tool schema through which S2 (AI Scientist-v2), S3 (AI-Researcher),
A0/A1/A4 and NLSS interact with GB1 and MADE (and, with an adapter, any
benchmark).  Design rules (§4):

1. ``get_task_dossier`` is byte-identical across compared systems.
2. No tool returns evaluator-only top-set thresholds, full labels, cardinality,
   or phase-diagram truth (hidden labels stay hidden).
3.-4. GB1/MADE reuse the native candidate space + oracle + budget.
5. Invalid/duplicate proposals consume the proposal/tool budget.
7. No baseline is forced to serialize an NLSS state.
8. Discovery score freezes at the terminal budget; later writing is post-freeze
   artifact.

This is the ADAPTER BOUNDARY for external AI-scientist systems: they may
translate an experiment action into a ``submit_batch`` call, but may NOT replace
their planning loop with NLSS/BO.  A ``toy_oracle`` is used for smoke; real
benchmarks inject the official oracle.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass
class CampaignBudget:
    query_cap: int = 40
    round_cap: int = 8
    batch: int = 4
    proposal_cap: int = 200  # invalid/duplicate proposals also counted (§4.5)


class ScientificCampaign:
    """One benchmark campaign session behind the §4 tool schema."""

    def __init__(
        self,
        adapter,
        *,
        budget: CampaignBudget | None = None,
        seed: int = 0,
        run_dir: str | Path | None = None,
        oracle: Callable[[Any, int], float] | None = None,
        toy_ok: bool = False,
    ) -> None:
        self.adapter = adapter
        self.budget = budget or CampaignBudget()
        self.seed = seed
        self.run_dir = Path(run_dir) if run_dir else None
        self._oracle = oracle
        self._toy_ok = bool(toy_ok)
        self._observations: dict[str, Any] = {}
        self._proposals_used = 0
        self._queries_used = 0
        self._t0 = time.time()
        # frozen token: every caller sees the same dossier string
        self._dossier = self._build_dossier()
        self._dossier_fingerprint = hashlib_sha256_dumps(self._dossier)

    # -- §4 tools --------------------------------------------------------

    def get_task_dossier(self) -> dict[str, Any]:
        """Public, byte-identical task dossier (NO hidden label/threshold/truth)."""
        return dict(self._dossier)

    def get_dossier_fingerprint(self) -> str:
        return self._dossier_fingerprint

    def get_observations(self) -> list[dict[str, Any]]:
        """Only queried observations; never the hidden true top-set/threshold."""
        return [dict(v) for v in sorted(self._observations.values(), key=lambda o: o["object_id"])]

    def get_candidate_space(self) -> list[str]:
        """Legal candidate ids (or generator handle) from the adapter surface."""
        return self._candidate_space()

    def submit_batch(
        self,
        candidate_ids: Sequence[str],
        hypotheses: Sequence[str] | None = None,
        rationale: str | None = None,
    ) -> list[dict[str, Any]]:
        """Propose candidates; charge proposal + query budget; return observations.

        Invalid/duplicate candidates consume the proposal cap (§4.5).  Every
        charge is recorded so resource accounting is exact.
        """
        cs = set(self.get_candidate_space())
        new_batch: list[dict[str, Any]] = []
        for i, cid in enumerate(candidate_ids[: self.budget.batch]):
            # authoritative budget: stop querying when capped (§4 identical-budget)
            if self._queries_used >= self.budget.query_cap:
                break
            if self._queries_used // max(1, self.budget.batch) >= self.budget.round_cap:
                break
            # single counter, charged for EVERY attempt (valid/invalid/duplicate, §4.5)
            self._proposals_used += 1
            if self._proposals_used > self.budget.proposal_cap:
                break
            if cid not in cs or cid in self._observations or self._hop(cid) is None:
                continue  # invalid/duplicate/unknown -> proposal charged, NO oracle query
            obj = self._hop(cid)
            rnd = self._queries_used // max(1, self.budget.batch) + 1
            val = self._oracle(self.adapter, obj, rnd) if self._oracle else 0.0
            rec = {"object_id": cid, "value": float(val), "round_index": rnd,
                   "rationale": rationale, "hypothesis": (hypotheses[i] if hypotheses and i < len(hypotheses) else None)}
            self._observations[cid] = rec
            self._queries_used += 1
            new_batch.append(rec)
        if self.run_dir is not None:
            self.save_artifact("observations", self.get_observations())
        return new_batch

    def remaining_budget(self) -> dict[str, Any]:
        return {
            "queries_remaining": max(0, self.budget.query_cap - self._queries_used),
            "proposals_remaining": max(0, self.budget.proposal_cap - self._proposals_used),
            "rounds_remaining": max(0, self.budget.round_cap - self._queries_used // max(1, self.budget.batch)),
        }

    def save_artifact(self, kind: str, payload: Any) -> str:
        """Immutable run artifact under run_dir; never overwrites, never re-read
        into the planning loop (post-freeze writing cannot change decisions)."""
        if self.run_dir is None:
            return ""
        rnd = self._queries_used // max(1, self.budget.batch) + 1
        rel = f"campaign/{kind}_r{rnd:02d}_q{self._queries_used:04d}.json"
        p = self.run_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            return str(p)  # immutable: no overwrite
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return str(p)

    # -- internals --------------------------------------------------------

    def _build_dossier(self) -> dict[str, Any]:
        try:
            dims = dict(self.adapter.attribute_domains())
        except Exception:
            dims = {}
        return {
            "task_id": self.adapter.task_id,
            "object_type": self.adapter.object_type,
            "attribute_domains": {k: (v if isinstance(v, (int, str, bool)) else "present") for k, v in dims.items()},
            "budget": {"queries": self.budget.query_cap, "rounds": self.budget.round_cap},
            "seed": self.seed,
            # NOTE: deliberately NO hidden labels / top-set threshold / cardinality / phase truth.
        }

    def _candidate_space(self) -> list[str]:
        if hasattr(self.adapter, "candidate_space"):
            try:
                return [str(x) for x in self.adapter.candidate_space()]
            except Exception:
                pass
        if not self._toy_ok:
            raise RuntimeError(
                "adapter exposes no candidate_space() and toy_ok=False — refusing to "
                "invent phantom candidates for a real benchmark (§4 rule 2).")
        # SMOKE ONLY: never used for a real benchmark.
        return [f"h{i}" for i in range(12)]

    def _hop(self, cid: str):
        """Turn a candidate id into a neutral grounded object for evaluation.

        The candidate id is the §4 boundary item; each system's executor maps it
        to the actual experiment.  A deterministic neutral GroundedObject keeps
        the boundary adapter-agnostic (real GB1/MADE oracles consume it)."""
        from ..core.types import GroundedObject
        return GroundedObject(object_id=cid, task_id=self.adapter.task_id,
                              object_type=self.adapter.object_type,
                              canonical_form=cid, payload={"text": cid},
                              display_text=cid, source_hypothesis_ids=())


def hashlib_sha256_dumps(obj) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]
