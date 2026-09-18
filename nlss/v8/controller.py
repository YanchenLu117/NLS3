"""V8 P0-c — frozen solution-field controller (Detail §3.4).

Primary controller score (P0 freezes all weights, normalizations, tie-breaking,
batch-diversity rule, and unavailable-candidate handling):

    a_t(z) = lambda_s * p_sat(z) + lambda_u * U_t(z)
           + lambda_c * CoverageGap_t(z) + lambda_o * OpenGap_t(z)

``NLSS-State`` does NOT use this controller: the host policy receives only the
frozen readout schema (see READOUT_ONLY_POLICY).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

READOUT_ONLY_POLICY = "nlss_state_uses_frozen_readout_schema_no_controller"

ControllerNormalizer = Callable[[float], float]


@dataclass(frozen=True)
class ControllerWeights:
    """P0-frozen weights; all four must be present and finite."""

    lambda_s: float
    lambda_u: float
    lambda_c: float
    lambda_o: float


@dataclass(frozen=True)
class ControllerDecision:
    scores: Mapping[str, float]
    batch: tuple[str, ...]
    skipped_unavailable: tuple[str, ...]
    tie_breaks: Mapping[str, tuple[str, ...]]
    diversity_rule_id: str


class FrozenController:
    def __init__(
        self,
        weights: ControllerWeights,
        *,
        normalizers: Mapping[str, ControllerNormalizer] | None = None,
        tie_break: str = "candidate_id_ascending",
        diversity_rule: Callable[[Sequence[str], Mapping[str, float]], list[str]] | None = None,
        diversity_rule_id: str = "none",
    ) -> None:
        for name in ("lambda_s", "lambda_u", "lambda_c", "lambda_o"):
            value = getattr(weights, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value:
                raise ValueError(f"weight {name} must be a finite number")
        self._w = weights
        self._normalizers = dict(normalizers or {})
        self._tie_break = tie_break
        self._diversity = diversity_rule
        self._diversity_rule_id = diversity_rule_id

    def score(self, candidates: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        """Score every available candidate; unavailable (missing/None term)
        candidates are skipped and reported by :meth:`select_batch`."""
        scores: dict[str, float] = {}
        for cid, terms in candidates.items():
            try:
                p_sat = self._term("p_sat", terms)
                utility = self._term("utility", terms)
                coverage_gap = self._term("coverage_gap", terms)
                open_gap = self._term("open_gap", terms)
            except KeyError:
                continue
            scores[cid] = (
                self._w.lambda_s * p_sat
                + self._w.lambda_u * utility
                + self._w.lambda_c * coverage_gap
                + self._w.lambda_o * open_gap
            )
        return scores

    def _term(self, name: str, terms: Mapping[str, float]) -> float:
        value = terms.get(name)
        if value is None:
            raise KeyError(name)
        value = float(value)
        normalizer = self._normalizers.get(name)
        return float(normalizer(value)) if normalizer is not None else value

    def select_batch(
        self,
        scores: Mapping[str, float],
        batch_size: int,
        *,
        unavailable: Sequence[str] = (),
    ) -> ControllerDecision:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        unavailable_set = set(unavailable)
        available = {cid: s for cid, s in scores.items() if cid not in unavailable_set}
        # deterministic order: score desc, then candidate id ascending
        ranked = sorted(available.items(), key=lambda kv: (-kv[1], kv[0]))
        ordered = [cid for cid, _ in ranked]
        if self._diversity is not None:
            ordered = list(self._diversity(ordered, dict(available)))
        batch = tuple(ordered[:batch_size])
        # tie-break reporting: groups of equal score within/below the batch cut
        ties: dict[str, tuple[str, ...]] = {}
        by_score: dict[float, list[str]] = {}
        for cid, s in ranked:
            by_score.setdefault(s, []).append(cid)
        for s, ids in by_score.items():
            if len(ids) > 1 and any(cid in batch for cid in ids):
                ties[str(s)] = tuple(sorted(ids))
        return ControllerDecision(
            scores=copy.deepcopy(dict(scores)),
            batch=batch,
            skipped_unavailable=tuple(sorted(unavailable_set & set(scores))),
            tie_breaks=ties,
            diversity_rule_id=self._diversity_rule_id,
        )
