"""Relational (revision-edge) metrics.

Phase 2 §2.6 asks whether the posterior assigns higher improvement probability to
revisions that actually improve the empirical utility.  These are pairwise
classification metrics evaluated over ``G_rev`` (or a revision-edge proxy in the
static bake-off): label = ``f(target) > f(source) + delta``, score = the
posterior's ``improvement_probability``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..core.types import PosteriorView


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve (Mann-Whitney U, pure numpy)."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    pos = s[y == 1.0]
    neg = s[y == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # n_pos * n_neg pairwise comparisons; count ties as 0.5
    total = len(pos) * len(neg)
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / total)


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (trapezoidal, pure numpy)."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(float(y.sum()), 1e-12)
    # prepend (recall=0, precision=1) and enforce non-increasing precision
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.trapezoid(precision, recall))


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mean squared error between predicted probability and binary label."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    if not mask.any():
        return float("nan")
    return float(np.mean((s[mask] - y[mask]) ** 2))


def log_loss(labels: Sequence[int], scores: Sequence[float], eps: float = 1e-12) -> float:
    """Binary cross-entropy of predicted probabilities against labels."""
    y = np.asarray(labels, dtype=float)
    s = np.clip(np.asarray(scores, dtype=float), eps, 1.0 - eps)
    mask = np.isfinite(y) & np.isfinite(s)
    if not mask.any():
        return float("nan")
    y, s = y[mask], s[mask]
    return float(-np.mean(y * np.log(s) + (1.0 - y) * np.log(1.0 - s)))


def pairwise_revision_metrics(
    posterior: PosteriorView,
    revision_edges: Sequence[tuple[str, str]],
    f_true: Mapping[str, float],
    *,
    delta: float = 0.0,
) -> Mapping[str, float]:
    """ROC-AUC / PR-AUC / Brier / log-loss over revision edges.

    ``revision_edges`` are ``(source_id, target_id)`` pairs.  The true label is 1
    when ``f_true[target] > f_true[source] + delta``; the score is the posterior's
    ``improvement_probability`` for that directed delta threshold.
    """
    labels: list[int] = []
    scores: list[float] = []
    for src, tgt in revision_edges:
        fs, ft = f_true.get(src), f_true.get(tgt)
        if fs is None or ft is None or not (np.isfinite(fs) and np.isfinite(ft)):
            continue
        labels.append(1 if ft > fs + delta else 0)
        scores.append(posterior.pairwise_delta(src, tgt, delta=delta).improvement_probability)
    return {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "brier": brier_score(labels, scores),
        "log_loss": log_loss(labels, scores),
        "n_edges": float(len(labels)),
    }
