"""Graph conversion helpers for recovery backends.

ScientificGraph / RevisionGraph are plain frozen containers; each recovery
backend converts them into its preferred numeric form (scipy sparse, torch,
NetworkX, PyG).  This module provides shared, dependency-light conversions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import ScientificGraph


def adjacency_matrix(
    graph: ScientificGraph,
    node_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Return a dense symmetric weighted adjacency matrix.

    ``node_ids`` fixes the row/column ordering; otherwise the graph's own order
    is used.
    """
    ids = list(node_ids) if node_ids is not None else list(graph.node_ids)
    index = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)
    adj = np.zeros((n, n), dtype=float)
    for edge in graph.edges:
        i = index.get(edge.source_id)
        j = index.get(edge.target_id)
        if i is None or j is None or i == j:
            continue
        adj[i, j] += float(edge.weight)
        adj[j, i] += float(edge.weight)
    return adj


def laplacian_matrix(
    graph: ScientificGraph,
    node_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Return the combinatorial graph Laplacian L = D - A."""
    adj = adjacency_matrix(graph, node_ids=node_ids)
    degree = adj.sum(axis=1)
    lap = np.diag(degree) - adj
    return (lap + lap.T) / 2.0


def sparse_laplacian(graph: ScientificGraph, node_ids: Sequence[str] | None = None):
    """Return a scipy CSR Laplacian (lazy scipy import)."""
    from scipy import sparse

    ids = list(node_ids) if node_ids is not None else list(graph.node_ids)
    index = {nid: i for i, nid in enumerate(ids)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for edge in graph.edges:
        i = index.get(edge.source_id)
        j = index.get(edge.target_id)
        if i is None or j is None or i == j:
            continue
        w = float(edge.weight)
        rows.extend((i, j))
        cols.extend((j, i))
        values.extend((w, w))
    adjacency = sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(ids), len(ids))
    ).tocsr()
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return sparse.diags(degree, format="csr") - adjacency
