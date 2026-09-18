"""Frozen V6 core dataclasses: language, grounding, relations, observations,
and posterior views.  No benchmark semantics live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Language layer
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LanguageHypothesis:
    """A natural-language hypothesis.  This is language-layer provenance, NOT a
    solution-space node."""

    hypothesis_id: str
    text: str

    rationale: str | None = None

    parent_object_ids: tuple[str, ...] = ()
    parent_hypothesis_ids: tuple[str, ...] = ()

    round_index: int = 0
    proposer: str = "agent"

    metadata: Mapping[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Grounding layer
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RawScientificObject:
    """P_tau(h) = x_tilde: a compiled, typed scientific specification before
    verification and canonicalization."""

    task_id: str
    object_type: str
    payload: Mapping[str, Any]

    source_hypothesis_id: str

    raw_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """V_tau: executability/domain/evaluator verification result."""

    valid: bool
    repaired_object: RawScientificObject | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroundedObject:
    """K_tau(x_tilde) = x: the canonical executable object — the NLSS node."""

    object_id: str
    task_id: str
    object_type: str
    canonical_form: str
    payload: Mapping[str, Any]
    display_text: str
    source_hypothesis_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Observation:
    """One revealed scalar empirical utility.  V6 keeps a single scalar field."""

    object_id: str
    value: float
    round_index: int

    noise_std: float | None = None
    evaluator: str = "benchmark"
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ScientificEdge:
    """G_sci edge: frozen, outcome-blind, domain-defined similarity."""

    edge_id: str
    source_id: str
    target_id: str
    weight: float
    distance: float

    relation_type: str = "scientific_similarity"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RevisionEdge:
    """G_rev edge: a meaningful executable scientific revision x_i -> x_j."""

    edge_id: str
    source_id: str
    target_id: str
    action_type: str
    action_description: str

    edit_payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScientificGraph:
    node_ids: tuple[str, ...]
    edges: tuple[ScientificEdge, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RevisionGraph:
    node_ids: tuple[str, ...]
    edges: tuple[RevisionEdge, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Posterior
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MarginalPosterior:
    object_id: str
    mean: float
    std: float
    solution_probability: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PairwiseDeltaPosterior:
    source_id: str
    target_id: str
    mean_delta: float
    std_delta: float
    improvement_probability: float
    threshold_delta: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PosteriorView(Protocol):
    """The estimator-agnostic posterior interface every recovery backend exposes."""

    def marginal(self, object_id: str) -> MarginalPosterior: ...

    def pairwise_delta(
        self,
        source_id: str,
        target_id: str,
        delta: float = 0.0,
    ) -> PairwiseDeltaPosterior: ...

    def sample_joint(
        self,
        object_ids: Sequence[str],
        n_samples: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return samples of shape [n_samples, len(object_ids)]."""
        ...
