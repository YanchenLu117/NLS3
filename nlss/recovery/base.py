"""Estimator-agnostic recovery contract (V6).

A RecoveryBackend maps (G_sci, D) -> Q(F_V): a joint empirical belief over the
whole constructed solution space.  Graph-Matérn GP, Laplacian smoothing,
RBF GP, multi-scale diffusion fields, and neural graph ensembles are all
candidate implementations behind this same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..core.types import (
    GroundedObject,
    Observation,
    PosteriorView,
    ScientificGraph,
)


@dataclass(frozen=True, slots=True)
class RecoveryInput:
    objects: tuple[GroundedObject, ...]
    scientific_graph: ScientificGraph
    observations: tuple[Observation, ...]
    solution_threshold: float | None = None
    # Optional outcome-blind scientific feature vectors (object_id -> array).
    # Required only by representation-based backends (e.g. Euclidean RBF GP);
    # graph-based backends ignore it.
    features: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RecoveryBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(self, recovery_input: RecoveryInput) -> None: ...

    @abstractmethod
    def posterior(self) -> PosteriorView: ...

    @abstractmethod
    def diagnostics(self) -> Mapping[str, Any]: ...

    def fit_posterior(self, recovery_input: RecoveryInput) -> PosteriorView:
        self.fit(recovery_input)
        return self.posterior()
