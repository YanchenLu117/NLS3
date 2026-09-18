"""The frozen TaskAdapter interface — the boundary between benchmark-specific
semantics (adapters) and NLSS semantics (core).

Every benchmark coding agent implements this.  ``build_scientific_graph``
deliberately does not accept observations, making G_sci outcome-blind from the
API level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TYPE_CHECKING, Protocol

from .types import (
    GroundedObject,
    LanguageHypothesis,
    Observation,
    RawScientificObject,
    RevisionGraph,
    ScientificGraph,
    VerificationResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..llm.base import LLMProvider


class TaskAdapter(ABC):
    @property
    @abstractmethod
    def task_id(self) -> str: ...

    @property
    @abstractmethod
    def object_type(self) -> str: ...

    @abstractmethod
    async def compile(
        self,
        hypothesis: LanguageHypothesis,
        llm: "LLMProvider",
    ) -> RawScientificObject:
        """P_tau: natural language -> typed executable specification."""

    @abstractmethod
    def verify(self, raw_object: RawScientificObject) -> VerificationResult:
        """V_tau: deterministic executability/domain/evaluator verification."""

    @abstractmethod
    def canonicalize(self, raw_object: RawScientificObject) -> GroundedObject:
        """K_tau: deterministic, idempotent, stable canonicalization."""

    @abstractmethod
    def build_scientific_graph(
        self,
        objects: tuple[GroundedObject, ...],
    ) -> ScientificGraph:
        """G_sci: frozen, outcome-blind, domain-defined relations."""

    @abstractmethod
    def build_revision_graph(
        self,
        objects: tuple[GroundedObject, ...],
    ) -> RevisionGraph:
        """G_rev: meaningful executable scientific revisions."""

    @abstractmethod
    async def evaluate(self, obj: GroundedObject) -> Observation:
        """Benchmark evaluator / oracle / surrogate."""

    @abstractmethod
    def render_object(self, obj: GroundedObject) -> str:
        """Human/agent-readable representation of a grounded object."""

    def feature_vectors(
        self,
        objects: tuple[GroundedObject, ...],
    ) -> "Mapping[str, Any] | None":
        """Explicit features contract (P0-1b) for Euclidean recovery backends.

        Returns an outcome-blind ``object_id -> feature-array`` map when the
        adapter can provide one (e.g. MADE's SOAP/composition vectors), else
        ``None``.  Declared as a concrete method on the interface so callers can
        rely on it without duck-typing via ``hasattr``; adapters opt into real
        features by overriding it.  The default (no features) lets the facade
        select a graph backend that genuinely fits.
        """
        return None
