"""Field-assumption diagnostics and solution-space recovery metrics.

Diagnostics (Phase 2 §2.3) test whether a frozen outcome-blind scientific graph
carries predictive signal for the empirical utility ``f``.  Recovery metrics
(§2.6) measure how well a posterior recovers the top solutions.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..core.graph import adjacency_matrix, laplacian_matrix


def local_smoothness(f: Mapping[str, float], graph, node_ids: Sequence[str] | None = None) -> float:
    """rho_neigh = corr(f_i, mean(f_j) over neighbors j in N(i))."""
    ids = list(node_ids) if node_ids is not None else list(graph.node_ids)
    adj = adjacency_matrix(graph, node_ids=ids)
    fv = np.asarray([f.get(nid, np.nan) for nid in ids], dtype=float)
    xs, ys = [], []
    for i in range(len(ids)):
        nbr = np.nonzero(adj[i])[0]
        if len(nbr) == 0 or not np.isfinite(fv[i]):
            continue
        nbr_vals = fv[nbr][np.isfinite(fv[nbr])]
        if len(nbr_vals) == 0:
            continue
        xs.append(fv[i])
        ys.append(float(np.mean(nbr_vals)))
    if len(xs) < 3:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def normalized_dirichlet_energy(f: Mapping[str, float], graph, node_ids: Sequence[str] | None = None) -> float:
    """E_G(f) = (f^T L f) / (f^T D f)."""
    ids = list(node_ids) if node_ids is not None else list(graph.node_ids)
    lap = laplacian_matrix(graph, node_ids=ids)
    adj = adjacency_matrix(graph, node_ids=ids)
    degree = adj.sum(axis=1)
    fv = np.asarray([f.get(nid, 0.0) for nid in ids], dtype=float)
    numerator = float(fv @ lap @ fv)
    denominator = float(fv @ (degree * fv))
    return numerator / denominator if denominator > 0 else float("nan")


def top_region_enrichment(
    f: Mapping[str, float],
    graph,
    node_ids: Sequence[str] | None = None,
    top_frac: float = 0.10,
) -> float:
    """P(j in Top-k | j in N(i), i in Top-k) / global Top-k fraction."""
    ids = list(node_ids) if node_ids is not None else list(graph.node_ids)
    adj = adjacency_matrix(graph, node_ids=ids)
    fv = np.asarray([f.get(nid, np.nan) for nid in ids], dtype=float)
    valid = np.isfinite(fv)
    k = max(1, int(np.ceil(valid.sum() * top_frac)))
    order = np.argsort(fv)
    order = order[np.isfinite(fv[order])]
    top = set(order[-k:].tolist())
    hit, total = 0, 0
    for i in top:
        for j in np.nonzero(adj[i])[0]:
            if j != i:
                total += 1
                if j in top:
                    hit += 1
    if total == 0:
        return float("nan")
    return (hit / total) / top_frac


def top_k_recall(
    f_true: Mapping[str, float],
    f_pred: Mapping[str, float],
    *,
    top_frac: float = 0.10,
) -> float:
    """Fraction of the true top-k solutions recovered in the predicted top-k."""
    ids = [nid for nid in f_true if nid in f_pred]
    if not ids:
        return float("nan")
    k = max(1, int(np.ceil(len(ids) * top_frac)))
    true_top = set(sorted(ids, key=lambda n: f_true[n], reverse=True)[:k])
    pred_top = set(sorted(ids, key=lambda n: f_pred[n], reverse=True)[:k])
    return len(true_top & pred_top) / len(true_top)


def top_k_auprc(
    f_true: Mapping[str, float],
    f_pred: Mapping[str, float],
    *,
    top_frac: float = 0.10,
) -> float:
    """PR-AUC of "is in true top-k" against the predicted ranking (higher = better)."""
    from .relational import pr_auc

    ids = [nid for nid in f_true if nid in f_pred]
    if not ids:
        return float("nan")
    k = max(1, int(np.ceil(len(ids) * top_frac)))
    true_top = set(sorted(ids, key=lambda n: f_true[n], reverse=True)[:k])
    labels = [1 if n in true_top else 0 for n in ids]
    scores = [f_pred[n] for n in ids]
    return pr_auc(labels, scores)


def spearman_rank(f_true: Mapping[str, float], f_pred: Mapping[str, float]) -> float:
    ids = [nid for nid in f_true if nid in f_pred]
    if len(ids) < 2:
        return float("nan")
    a = np.asarray([f_true[n] for n in ids])
    b = np.asarray([f_pred[n] for n in ids])
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def rmse(f_true: Mapping[str, float], f_pred: Mapping[str, float]) -> float:
    ids = [nid for nid in f_true if nid in f_pred]
    if not ids:
        return float("nan")
    a = np.asarray([f_true[n] for n in ids])
    b = np.asarray([f_pred[n] for n in ids])
    return float(np.sqrt(np.mean((a - b) ** 2)))


def aurc(recovery_by_budget) -> float:
    """Area Under the Recovery-vs-Budget curve (AURC), trapezoidal integral.

    ``recovery_by_budget`` may be:
      * a sequence of recovery amounts at successive unit budget steps
        (index i -> budget i), recovery magnitude [[0, 1]]; or
      * a Mapping ``{budget: recovery}``; or
      * an iterable of ``(budget, recovery)`` pairs.

    Recovery amounts are cumulative (monotonically non-decreasing in budget):
    AURC = (1 / (Bmax - Bmin)) * INT_{Bmin}^{Bmax} recovery(B) dB, i.e. the
    unit-normalized trapezoidal AUC in [[0, 1]].  Returns 0.0 when fewer than
    two distinct budget points are available.
    """
    if isinstance(recovery_by_budget, Mapping):
        pairs = sorted((float(b), float(r)) for b, r in recovery_by_budget.items())
    else:
        items = list(recovery_by_budget)
        if items and not isinstance(items[0], (tuple, list)):
            pairs = [(float(i), float(r)) for i, r in enumerate(items)]
        else:
            pairs = [(float(b), float(r)) for b, r in items]
    pairs = sorted(pairs)
    if len(pairs) < 2 or pairs[-1][0] <= pairs[0][0]:
        return 0.0
    bmin, bmax = pairs[0][0], pairs[-1][0]
    area = 0.0
    for (b0, r0), (b1, r1) in zip(pairs, pairs[1:]):
        if b1 <= b0:
            continue
        area += 0.5 * (r0 + r1) * (b1 - b0)
    return float(area / (bmax - bmin))
