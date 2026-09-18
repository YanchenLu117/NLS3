"""Solution-Space Reconstruction (Agent D) — read-only consumer layer.

Provides schema / exporter / metrics / render for reconstructing a solution
space from a shared state + posterior + adapter.  Does NOT modify
core/recovery/EvidenceBuilder/StateStrategy.
"""

from __future__ import annotations

from .schema import (
    RecoveredSolutionNode,
    RecoveredRegion,
    RecoveredBoundary,
    RecoveredFrontier,
)
from .exporter import SolutionSpaceExporter, _simplex_xy, _projection
from .metrics import (
    region_recall,
    boundary_error,
    frontier_coverage,
    uncovered_distance,
    component_recall,
    coverage_entropy,
    reconstruction_metrics,
)
from .render import render_map

__all__ = [
    "RecoveredSolutionNode",
    "RecoveredRegion",
    "RecoveredBoundary",
    "RecoveredFrontier",
    "SolutionSpaceExporter",
    "_simplex_xy",
    "_projection",
    "region_recall",
    "boundary_error",
    "frontier_coverage",
    "uncovered_distance",
    "component_recall",
    "coverage_entropy",
    "reconstruction_metrics",
    "render_map",
]
