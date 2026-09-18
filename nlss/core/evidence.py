"""Evidence dataclasses and the shared EvidenceBuilder.

The EvidenceBuilder is Core.  Benchmark agents must not rewrite evidence
ranking logic — they only provide the TaskAdapter (render + relations) and the
shared SolutionSpaceState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from .errors import RecoveryNotFitted
from .types import GroundedObject

if TYPE_CHECKING:  # pragma: no cover
    from .interfaces import TaskAdapter
    from .state import SolutionSpaceState


@dataclass(frozen=True, slots=True)
class AnchorEvidence:
    object_id: str
    representation: str
    mean_utility: float | None
    uncertainty: float | None
    solution_probability: float | None
    observed: bool


@dataclass(frozen=True, slots=True)
class AlternativeEvidence:
    object_id: str
    representation: str
    region_id: str | None
    mean_utility: float | None
    relation_to_anchor: str | None


@dataclass(frozen=True, slots=True)
class BoundaryEvidence:
    object_id: str
    representation: str
    uncertainty: float
    explanation: str


@dataclass(frozen=True, slots=True)
class RevisionEvidence:
    source_id: str
    target_id: str
    source_repr: str
    target_repr: str
    action_type: str
    action_description: str
    mean_delta: float | None
    std_delta: float | None
    improvement_probability: float | None
    confidence_label: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    task_id: str
    round_index: int
    anchors: tuple[AnchorEvidence, ...]
    alternatives: tuple[AlternativeEvidence, ...]
    boundaries: tuple[BoundaryEvidence, ...]
    revisions: tuple[RevisionEvidence, ...]
    observed_count: int
    constructed_count: int
    state_token_budget: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EvidenceBuilder:
    """Frozen first-version evidence ranking logic."""

    def __init__(
        self,
        *,
        n_anchors: int = 4,
        n_alternatives: int = 3,
        n_boundaries: int = 3,
        n_positive_revisions: int = 6,
        n_uncertain_revisions: int = 3,
        n_negative_revisions: int = 3,
        token_budget: int = 1800,
    ) -> None:
        self.n_anchors = n_anchors
        self.n_alternatives = n_alternatives
        self.n_boundaries = n_boundaries
        self.n_positive_revisions = n_positive_revisions
        self.n_uncertain_revisions = n_uncertain_revisions
        self.n_negative_revisions = n_negative_revisions
        self.token_budget = token_budget

    def build(
        self,
        state: "SolutionSpaceState",
        adapter: "TaskAdapter",
    ) -> EvidencePacket:
        if state.posterior is None:
            raise RecoveryNotFitted("posterior must be fitted before building evidence")

        objects = state.objects
        obs = state.observations

        # marginal posteriors
        marginals: dict[str, Any] = {}
        for oid in objects:
            marginals[oid] = state.posterior.marginal(oid)

        def _mean(oid: str) -> float:
            return float(marginals[oid].mean)

        def _std(oid: str) -> float:
            return float(marginals[oid].std)

        def _sol_prob(oid: str) -> float | None:
            return marginals[oid].solution_probability

        # ---- anchors: high E[f], spread across regions (greedy diversity) ----
        ordered = sorted(objects, key=lambda o: _mean(o), reverse=True)
        anchors: list[AnchorEvidence] = []
        for oid in ordered:
            if len(anchors) >= self.n_anchors:
                break
            anchors.append(
                AnchorEvidence(
                    object_id=oid,
                    representation=adapter.render_object(objects[oid]),
                    mean_utility=_mean(oid),
                    uncertainty=_std(oid),
                    solution_probability=_sol_prob(oid),
                    observed=state.is_observed(oid),
                )
            )

        # ---- boundaries: high uncertainty or p_sol ~= 0.5 ----
        boundary_candidates = sorted(
            objects,
            key=lambda o: (
                _sol_prob(o) is not None and abs(_sol_prob(o) - 0.5) < 0.25,
                _std(o),
            ),
            reverse=True,
        )
        boundaries: list[BoundaryEvidence] = []
        for oid in boundary_candidates:
            if len(boundaries) >= self.n_boundaries:
                break
            boundaries.append(
                BoundaryEvidence(
                    object_id=oid,
                    representation=adapter.render_object(objects[oid]),
                    uncertainty=_std(oid),
                    explanation="high posterior uncertainty or p_sol near 0.5",
                )
            )

        # ---- alternatives: high-quality representatives from distinct regions ----
        # simple diversity proxy: pick high-mean unobserved nodes not adjacent to anchors
        alternatives: list[AlternativeEvidence] = []
        for oid in ordered:
            if len(alternatives) >= self.n_alternatives:
                break
            if oid in {a.object_id for a in anchors}:
                continue
            alternatives.append(
                AlternativeEvidence(
                    object_id=oid,
                    representation=adapter.render_object(objects[oid]),
                    region_id=None,
                    mean_utility=_mean(oid),
                    relation_to_anchor=None,
                )
            )

        # ---- revisions: high q_ij / high uncertainty / low q_ij ----
        revs: list[RevisionEvidence] = []
        for edge in state.revision_graph.edges:
            if edge.source_id not in objects or edge.target_id not in objects:
                continue
            try:
                delta = state.posterior.pairwise_delta(edge.source_id, edge.target_id)
            except Exception:
                continue
            revs.append(
                RevisionEvidence(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    source_repr=adapter.render_object(objects[edge.source_id]),
                    target_repr=adapter.render_object(objects[edge.target_id]),
                    action_type=edge.action_type,
                    action_description=edge.action_description,
                    mean_delta=delta.mean_delta,
                    std_delta=delta.std_delta,
                    improvement_probability=delta.improvement_probability,
                    confidence_label=None,
                    metadata=dict(edge.metadata),
                )
            )

        positives = sorted(
            revs, key=lambda r: r.improvement_probability or -1.0, reverse=True
        )[: self.n_positive_revisions]
        negatives = sorted(
            revs, key=lambda r: r.improvement_probability or 1.0
        )[: self.n_negative_revisions]
        uncertain = sorted(
            revs, key=lambda r: r.std_delta or -1.0, reverse=True
        )[: self.n_uncertain_revisions]

        return EvidencePacket(
            task_id=state.task_id,
            round_index=state.round_index,
            anchors=tuple(anchors),
            alternatives=tuple(alternatives),
            boundaries=tuple(boundaries),
            revisions=tuple(positives + uncertain + negatives),
            observed_count=len(obs),
            constructed_count=len(objects),
            state_token_budget=self.token_budget,
        )
