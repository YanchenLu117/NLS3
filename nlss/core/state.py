"""SolutionSpaceState — the V6 persistent scientific state.

S_t = (V_t, G_sci, G_rev, Q_t, L_t), where trajectory T_t is provenance/history
rather than scientific state.  Constructed nodes V_t may exist without
observations (|O_t| < |V_t|).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .types import (
    GroundedObject,
    LanguageHypothesis,
    Observation,
    PosteriorView,
    RevisionGraph,
    ScientificGraph,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..recovery.base import RecoveryBackend


@dataclass(slots=True)
class SolutionSpaceState:
    task_id: str
    round_index: int

    objects: dict[str, GroundedObject]
    hypotheses: dict[str, LanguageHypothesis]

    scientific_graph: ScientificGraph
    revision_graph: RevisionGraph

    observations: dict[str, Observation]

    posterior: PosteriorView | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def constructed_ids(self) -> tuple[str, ...]:
        return tuple(self.objects)

    @property
    def observed_ids(self) -> tuple[str, ...]:
        return tuple(self.observations)

    def is_observed(self, object_id: str) -> bool:
        return object_id in self.observations
