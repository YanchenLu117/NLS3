"""Solution-Space Reconstruction — exporter (Agent D, read-only consumer).

Dumps a reconstructed solution-space map from a shared SolutionSpaceState +
posterior + TaskAdapter into parquet artifacts:
  nodes.parquet / scientific_edges.parquet / revision_edges.parquet /
  regions.parquet / boundaries.parquet / frontiers.parquet
plus JSON snapshots per round.

This is a consumer layer: it does NOT modify core/recovery/EvidenceBuilder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import (
    RecoveredSolutionNode,
    RecoveredRegion,
    RecoveredBoundary,
    RecoveredFrontier,
)

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


# ---------------------------------------------------------------------------
# 2D projection helpers (composition-simplex for MADE; generic fallback)
# ---------------------------------------------------------------------------
def _simplex_xy(frac: Mapping[str, float], elements: Sequence[str]) -> tuple[float, float]:
    """Map a composition-fraction vector onto a 2-simplex (x,y) in [0,1]x[0,1].
    For n>3 elements we use the first three fractions normalized (pseudo-ternary),
    which is the standard MADE ternary visualization."""
    if len(elements) == 0:
        return 0.0, 0.0
    f = np.array([frac.get(el, 0.0) for el in elements], dtype=float)
    if f.sum() <= 0:
        return 0.0, 0.0
    f = f / f.sum()
    if len(f) >= 3:
        a, b, c = f[0], f[1], f[2]
    elif len(f) == 2:
        a, b, c = f[0], f[1], 0.0
    else:
        a, b, c = f[0], 0.0, 0.0
    # barycentric -> cartesian on equilateral triangle
    x = 0.5 * (2.0 * b + c) / (a + b + c + 1e-9)
    y = (math.sqrt(3.0) / 2.0) * c / (a + b + c + 1e-9)
    return float(x), float(y)


def _projection(state, adapter) -> dict[str, tuple[float, float]]:
    """Return object_id -> (x,y). Uses composition fractions when available."""
    out: dict[str, tuple[float, float]] = {}
    for oid, obj in state.objects.items():
        comp = obj.payload.get("composition") if hasattr(obj, "payload") else None
        if comp and hasattr(adapter, "elements"):
            els = list(adapter.elements)
            frac = {el: float(comp.get(el, 0)) for el in els}
            out[oid] = _simplex_xy(frac, els)
        else:
            out[oid] = (0.0, 0.0)
    return out


# ---------------------------------------------------------------------------
# Region / boundary / frontier recovery (MADE: joint-MC Bayes hull semantics)
# ---------------------------------------------------------------------------
def _posterior_marginals(state):
    """Return oid -> (mean, std, sol_prob) from state.posterior if fitted."""
    out: dict[str, tuple[float, float, float | None]] = {}
    if state.posterior is None:
        return out
    for oid in state.objects:
        try:
            m = state.posterior.marginal(oid)
            out[oid] = (float(m.mean), float(m.std), m.solution_probability)
        except Exception:
            continue
    return out


def _recover_regions(state, adapter, nodes, *, eps=0.1, n_clusters=None):
    """Simple deterministic region recovery.

    For MADE, a region is a stable-composition hull region: nodes whose
    posterior solution probability / utility is in the top stable band, grouped
    by connected components of the scientific graph.  This is a consumer-side
    approximation; benchmark ground truth is only injected via metrics.
    """
    import networkx as nx

    marg = _posterior_marginals(state)
    # stable band: sol_prob >= 0.5 or utility >= posterior median (fallback)
    scores = []
    for n in nodes:
        mm = marg.get(n.object_id)
        if mm is None:
            scores.append(0.0)
        else:
            sol = mm[2] if mm[2] is not None else 0.0
            scores.append(sol)
    thr = float(np.median(scores)) if scores else 0.0
    member_ids = [n.object_id for n, s in zip(nodes, scores) if s >= max(0.5, thr)]

    g = nx.Graph()
    for n in nodes:
        g.add_node(n.object_id)
    for e in state.scientific_graph.edges:
        g.add_edge(e.source_id, e.target_id)
    comps = [sorted(c) for c in nx.connected_components(g) if any(x in member_ids for x in c)]
    regions = []
    for i, comp in enumerate(comps):
        comp_set = set(comp)
        region_nodes = [n for n in nodes if n.object_id in comp_set and n.object_id in member_ids]
        if not region_nodes:
            continue
        cx = float(np.mean([n.x for n in region_nodes]))
        cy = float(np.mean([n.y for n in region_nodes]))
        best = max((marg.get(n.object_id, (0, 0, 0))[0] for n in region_nodes), default=0.0)
        regions.append(RecoveredRegion(
            region_id=f"R{i}",
            center_x=cx, center_y=cy, size=float(len(region_nodes)),
            nodes=tuple(n.object_id for n in region_nodes),
            best_utility=float(best),
            metadata={"n_members": len(region_nodes)},
        ))
    return regions


def _recover_boundaries(state, adapter, nodes, regions, *, eps=0.1):
    """Boundary = nodes adjacent to a stable region but not in it (transition)."""
    member = {oid for r in regions for oid in r.nodes}
    bound_ids = []
    for e in state.scientific_graph.edges:
        s_in = e.source_id in member
        t_in = e.target_id in member
        if s_in != t_in:
            other = e.target_id if s_in else e.source_id
            if other not in member:
                bound_ids.append(other)
    bound_ids = sorted(set(bound_ids))
    return [RecoveredBoundary(boundary_id=f"B{i}", node_ids=tuple(bound_ids[i:i + 1]),
                              threshold=eps, uncertainty=None)
            for i in range(0, len(bound_ids), max(1, len(bound_ids) // 8 + 1))][:8]


def _recover_frontiers(state, adapter, nodes, *, eps=0.1, top_k=8):
    """Frontier = unobserved nodes with highest posterior uncertainty."""
    marg = _posterior_marginals(state)
    scored = []
    for n in nodes:
        if n.observed:
            continue
        mm = marg.get(n.object_id)
        std = mm[1] if mm else 0.0
        scored.append((std, n.object_id))
    scored.sort(reverse=True)
    ids = [oid for _, oid in scored[:top_k]]
    if not ids:
        return []
    return [RecoveredFrontier(frontier_id="F0", node_ids=tuple(ids),
                              mean_uncertainty=float(np.mean([s for s, _ in scored[:top_k]])) if scored else 0.0,
                              coverage=None)]


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------
class SolutionSpaceExporter:
    """Read-only exporter of the reconstructed solution space."""

    def __init__(self, *, eps: float = 0.1):
        self.eps = eps

    def dump(self, state, adapter, outdir: str | Path) -> dict[str, Any]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)

        proj = _projection(state, adapter)
        marg = _posterior_marginals(state)

        # nodes
        nodes = []
        for oid, obj in state.objects.items():
            x, y = proj.get(oid, (0.0, 0.0))
            mm = marg.get(oid)
            nodes.append(RecoveredSolutionNode(
                object_id=oid,
                representation=adapter.render_object(obj) if hasattr(adapter, "render_object") else str(obj),
                x=x, y=y,
                mean_utility=mm[0] if mm else None,
                uncertainty=mm[1] if mm else None,
                solution_probability=mm[2] if mm else None,
                observed=oid in state.observations,
                round_index=state.observations[oid].round_index if oid in state.observations else None,
            ))
        nodes_by_id = {n.object_id: n for n in nodes}

        # regions / boundaries / frontiers
        regions = _recover_regions(state, adapter, nodes, eps=self.eps)
        boundaries = _recover_boundaries(state, adapter, nodes, regions, eps=self.eps)
        frontiers = _recover_frontiers(state, adapter, nodes, eps=self.eps)

        if pd is not None:
            pd.DataFrame([{
                "object_id": n.object_id, "representation": n.representation,
                "x": n.x, "y": n.y, "mean_utility": n.mean_utility,
                "uncertainty": n.uncertainty, "solution_probability": n.solution_probability,
                "observed": n.observed, "round_index": n.round_index, "region_id": n.region_id,
            } for n in nodes]).to_parquet(out / "nodes.parquet", index=False)

            def _edges(edge_iter, kind: str):
                rows = []
                for e in edge_iter:
                    rows.append({
                        "source_id": e.source_id, "target_id": e.target_id,
                        "kind": kind, "weight": getattr(e, "weight", None),
                        "distance": getattr(e, "distance", None),
                        "action_type": getattr(e, "action_type", None),
                        "action_description": getattr(e, "action_description", None),
                    })
                return pd.DataFrame(rows)
            _edges(state.scientific_graph.edges, "scientific").to_parquet(out / "scientific_edges.parquet", index=False)
            _edges(state.revision_graph.edges, "revision").to_parquet(out / "revision_edges.parquet", index=False)
            pd.DataFrame([{
                "region_id": r.region_id, "center_x": r.center_x, "center_y": r.center_y,
                "size": r.size, "nodes": list(r.nodes), "best_utility": r.best_utility,
            } for r in regions]).to_parquet(out / "regions.parquet", index=False)
            pd.DataFrame([{
                "boundary_id": b.boundary_id, "node_ids": list(b.node_ids),
                "threshold": b.threshold, "uncertainty": b.uncertainty,
            } for b in boundaries]).to_parquet(out / "boundaries.parquet", index=False)
            pd.DataFrame([{
                "frontier_id": f.frontier_id, "node_ids": list(f.node_ids),
                "mean_uncertainty": f.mean_uncertainty, "coverage": f.coverage,
            } for f in frontiers]).to_parquet(out / "frontiers.parquet", index=False)

        # snapshots per round
        snap_dir = out / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        rounds = sorted({n.round_index for n in nodes if n.round_index is not None})
        for r in rounds:
            snap = {
                "round": r,
                "nodes": [{
                    "object_id": n.object_id, "x": n.x, "y": n.y,
                    "mean_utility": n.mean_utility, "uncertainty": n.uncertainty,
                    "solution_probability": n.solution_probability,
                    "observed": n.observed and n.round_index is not None and n.round_index <= r,
                } for n in nodes if n.round_index is None or n.round_index <= r],
                "regions": [{"region_id": x.region_id, "center_x": x.center_x,
                             "center_y": x.center_y, "size": x.size} for x in regions],
            }
            (snap_dir / f"round_{r}.json").write_text(json.dumps(snap, indent=2))

        return {
            "n_nodes": len(nodes), "n_regions": len(regions),
            "n_boundaries": len(boundaries), "n_frontiers": len(frontiers),
            "outdir": str(out),
        }
