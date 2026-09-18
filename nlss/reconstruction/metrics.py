"""Solution-Space Reconstruction — metrics (Agent D, read-only consumer).

Benchmark ground truth is injected via a simple evaluator interface
(``ground_truth`` callable / object).  Core/recovery untouched.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def region_recall(recovered_regions: Sequence[Any],
                  true_region_ids: Sequence[str]) -> float:
    """Fraction of true solution regions that contain at least one recovered
    stable node.  ``true_region_ids`` is supplied by the benchmark ground truth."""
    rec = set(true_region_ids)
    if not rec:
        return 0.0
    hit = set()
    for r in recovered_regions:
        # region_id or any node id in the region
        if r.region_id in rec:
            hit.add(r.region_id)
        for nid in getattr(r, "nodes", ()):
            if nid in rec:
                hit.add(nid)
    return _safe_div(len(hit), len(rec))


def boundary_error(recovered_boundaries: Sequence[Any],
                   true_boundary_nodes: Sequence[str],
                   node_distance: Callable[[str, str], float] | None = None) -> float:
    """Mean distance from each true boundary node to the nearest recovered
    boundary node.  If no recovered boundaries, returns inf (bad)."""
    if not true_boundary_nodes:
        return 0.0
    rec_nodes = [nid for b in recovered_boundaries for nid in getattr(b, "node_ids", ())]
    if not rec_nodes:
        return float("inf")
    errs = []
    for t in true_boundary_nodes:
        if node_distance is None:
            # fallback: exact match only
            d = 0.0 if t in rec_nodes else 1.0
        else:
            d = min(node_distance(t, r) for r in rec_nodes)
        errs.append(d)
    return float(np.mean(errs))


def frontier_coverage(recovered_frontiers: Sequence[Any],
                      true_frontier_nodes: Sequence[str]) -> float:
    """Fraction of true frontier nodes that lie in a recovered frontier."""
    if not true_frontier_nodes:
        return 0.0
    rec = {nid for f in recovered_frontiers for nid in getattr(f, "node_ids", ())}
    return _safe_div(sum(1 for t in true_frontier_nodes if t in rec), len(true_frontier_nodes))


def uncovered_distance(recovered_nodes: Sequence[Any],
                       all_true_nodes: Sequence[str],
                       node_distance: Callable[[str, str], float] | None = None) -> float:
    """Mean distance from every true node to the nearest recovered node."""
    rec_ids = {n.object_id for n in recovered_nodes}
    if not rec_ids:
        return float("inf")
    if node_distance is None:
        return _safe_div(sum(1 for t in all_true_nodes if t not in rec_ids), len(all_true_nodes))
    ds = []
    for t in all_true_nodes:
        ds.append(min(node_distance(t, r) for r in rec_ids))
    return float(np.mean(ds))


def component_recall(recovered_regions: Sequence[Any],
                     true_components: Sequence[Sequence[str]]) -> float:
    """Fraction of true solution-space components touched by any recovered region."""
    if not true_components:
        return 0.0
    rec = {nid for r in recovered_regions for nid in getattr(r, "nodes", ())}
    hit = sum(1 for comp in true_components if any(c in rec for c in comp))
    return _safe_div(hit, len(true_components))


def coverage_entropy(recovered_regions: Sequence[Any], n_total: int) -> float:
    """Shannon entropy of region sizes (normalized by log n_regions)."""
    if n_total <= 0:
        return 0.0
    sizes = [max(1, getattr(r, "size", 0) or 1) for r in recovered_regions]
    if not sizes:
        return 0.0
    p = np.asarray(sizes, dtype=float) / sum(sizes)
    h = -float(np.sum(p * np.log(p)))
    return h / math.log(len(p)) if len(p) > 1 else 0.0


def reconstruction_metrics(recovered_regions, recovered_boundaries,
                           recovered_frontiers, recovered_nodes,
                           *, true_region_ids, true_boundary_nodes,
                           true_frontier_nodes, all_true_nodes,
                           true_components=None, node_distance=None,
                           n_total=None) -> Mapping[str, float]:
    return {
        "region_recall": region_recall(recovered_regions, true_region_ids),
        "boundary_error": boundary_error(recovered_boundaries, true_boundary_nodes, node_distance),
        "frontier_coverage": frontier_coverage(recovered_frontiers, true_frontier_nodes),
        "uncovered_distance": uncovered_distance(recovered_nodes, all_true_nodes, node_distance),
        "component_recall": component_recall(recovered_regions, true_components or []),
        "coverage_entropy": coverage_entropy(recovered_regions, n_total or len(recovered_nodes)),
    }
