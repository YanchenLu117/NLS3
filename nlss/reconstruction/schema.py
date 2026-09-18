"""Solution-Space Reconstruction — schema (Agent D, read-only consumer layer).

Frozen contract: this module only CONSUMES state + posterior + adapter.  It
must not import core semantics into core; it extends the shared evidence
dataclasses for reconstruction artifacts.

Alignment: total-doc section 2 — RecoveredSolutionNode / RecoveredRegion /
RecoveredBoundary / RecoveredFrontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RecoveredSolutionNode:
    """A node in the recovered solution-space map.

    ``x``/``y`` are a 2D projection (e.g. composition simplex / UMAP).  Utility
    and uncertainty come from the shared posterior (marginal mean/std).
    ``solution_probability`` is the posterior solution probability if exposed.
    """

    object_id: str
    representation: str
    x: float
    y: float
    mean_utility: float | None = None
    uncertainty: float | None = None
    solution_probability: float | None = None
    observed: bool = False
    round_index: int | None = None
    region_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveredRegion:
    """A recovered solution region (e.g. stable-composition hull region)."""

    region_id: str
    center_x: float
    center_y: float
    size: float  # number of member nodes (or area proxy)
    nodes: tuple[str, ...] = ()
    best_utility: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveredBoundary:
    """A boundary between stable and unstable recovered regions."""

    boundary_id: str
    node_ids: tuple[str, ...] = ()
    threshold: float | None = None
    uncertainty: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveredFrontier:
    """A high-uncertainty / under-covered frontier in the recovered map."""

    frontier_id: str
    node_ids: tuple[str, ...] = ()
    mean_uncertainty: float = 0.0
    coverage: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
