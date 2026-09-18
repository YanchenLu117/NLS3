"""Finite-oracle recovery-metric evaluator (NLSS V7).

Evaluates a recovery posterior over the **unqueried** candidates only, so a
method cannot get credit for memorizing queried labels (§9.9).  Implements the
recovery metric family:

  * UnseenRecall / UnseenPrecision / F1
  * PR-AUC for solution membership
  * RegionRecall (fraction of true high-yield connected components covered)
  * UncoveredDistance (min distance from an unrecovered solution to a recovered
    supported structure)
  * calibration (Brier score / ECE) for ``p_t(x)``
  * AURC (area under the recovery-vs-budget curve, §4.7)

All metrics accept an explicit ``node_features``/``candidate`` ground object so
geometry (region components, distances) stays **outcome-blind** and is the same
the recovery model sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


def pr_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Average precision (PR-AUC) for binary labels, higher = better."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(labels) < 2 or labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    if _HAS_SKLEARN:
        return float(average_precision_score(labels, scores))
    # fallback: rank-based average precision over true positives
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    pos_idx = np.where(ranked_labels == 1)[0]
    if len(pos_idx) == 0:
        return float("nan")
    prec_at = (np.arange(len(pos_idx)) + 1) / (pos_idx + 1)
    return float(np.mean(prec_at))


def _ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
    if len(labels) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    w = np.digitize(probs, bins) - 1
    w = np.clip(w, 0, n_bins - 1)
    tot = 0.0
    cnt = len(labels)
    for b in range(n_bins):
        m = w == b
        if not np.any(m):
            continue
        acc = float(np.mean(labels[m]))
        conf = float(np.mean(probs[m]))
        tot += (len(m) / cnt) * abs(acc - conf)
    return float(tot)


@dataclass(frozen=True, slots=True)
class RecoveryEvalResult:
    """All recovery metrics for one checkpoint."""

    n_unqueried: int
    unseen_recall: float
    unseen_precision: float
    unseen_f1: float
    pr_auc: float
    region_recall: float
    uncovered_distance: float | None
    brier: float
    ece: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_unqueried": self.n_unqueried,
            "unseen_recall": round(self.unseen_recall, 5),
            "unseen_precision": round(self.unseen_precision, 5),
            "unseen_f1": round(self.unseen_f1, 5),
            "pr_auc": round(self.pr_auc, 5),
            "region_recall": round(self.region_recall, 5),
            "uncovered_distance": (
                round(self.uncovered_distance, 5) if self.uncovered_distance is not None else None
            ),
            "brier": round(self.brier, 5),
            "ece": round(self.ece, 5),
        }


def evaluate_recovery(
    *,
    unqueried_candidates: Sequence[Any],
    true_membership: Callable[[Any], bool],
    pred_solution_prob: Callable[[Any], float],
    true_solution_prob: Callable[[Any], float] | None = None,
    node_features: Mapping[Any, np.ndarray] | None = None,
    true_components_of: Callable[[Any], frozenset] | None = None,
    recovered_supported: Sequence[Any] = (),
    region_rank_K: int | None = None,
) -> RecoveryEvalResult:
    """Evaluate recovery over unqueried candidates.

    Parameters
    ----------
    unqueried_candidates:
        The candidates NOT yet queried (evaluation pool, §9.9).
    true_membership:
        Outcome-based solution membership (uses gold — evaluator only).
    pred_solution_prob:
        The method's predicted P(sol | x) on a candidate.
    true_solution_prob:
        Optional indicator probability (1.0 if member) for calibration.
    node_features:
        Optional outcome-blind candidate -> feature vector, for region geometry.
    true_components_of:
        Optional outcome-blind component id (frozen geometry) per candidate, for
        RegionRecall.
    recovered_supported:
        Candidates the method has explicitly marked as supported solutions
        (for UncoveredDistance).
    """
    cands = list(unqueried_candidates)
    if not cands:
        return RecoveryEvalResult(
            n_unqueried=0, unseen_recall=0.0, unseen_precision=0.0, unseen_f1=0.0,
            pr_auc=float("nan"), region_recall=float("nan"),
            uncovered_distance=None, brier=float("nan"), ece=float("nan"),
        )

    labels = np.asarray([1.0 if true_membership(c) else 0.0 for c in cands])
    probs = np.asarray([float(pred_solution_prob(c)) for c in cands], dtype=float)
    n_pos = int(labels.sum())

    # Precision / Recall / F1 at the eta=0.5 high-probability cut.
    pred_pos = probs >= 0.5
    tp = int(np.sum(pred_pos & (labels == 1)))
    fp = int(np.sum(pred_pos & (labels == 0)))
    fn = int(np.sum((~pred_pos) & (labels == 1)))
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # RegionRecall: fraction of true solution components touched by the method's
    # recovery.  With region_rank_K set, "touched" uses the method's TOP-K ranked
    # (by pred prob) unqueried candidates -- a ranking-based, precision-immune
    # definition (report 9.9 "recovery criterion").  Without it, falls back to
    # the nominal p>=0.5 supported set.
    region_recall = float("nan")
    if true_components_of is not None:
        gold_comp = {true_components_of(c): c for c in cands if true_membership(c)}
        if region_rank_K is not None:
            order = np.argsort(-probs)
            touched = [cands[i] for i in order[: min(region_rank_K, len(cands))]]
        else:
            touched = list(recovered_supported)
        covered = {true_components_of(c) for c in touched if true_components_of(c) is not None}
        if gold_comp:
            hits = sum(1 for comp in gold_comp if comp in covered)
            region_recall = hits / len(gold_comp)

    # UncoveredDistance: minimum feature-distance from an unrecovered solution to
    # a recovered supported structure.
    uncovered_distance = None
    if node_features is not None and recovered_supported:
        sup_vecs = [node_features[c] for c in recovered_supported if c in node_features]
        miss = [c for c in cands if true_membership(c) and not (probs[cands.index(c)] >= 0.5)]
        if sup_vecs and miss:
            sup_arr = np.asarray(sup_vecs)
            dmin = float("inf")
            for c in miss:
                v = node_features.get(c)
                if v is None:
                    continue
                d = float(np.min(np.linalg.norm(sup_arr - np.asarray(v), axis=1)))
                dmin = min(dmin, d)
            uncovered_distance = float(dmin) if dmin != float("inf") else None

    # PR-AUC of solution membership against predicted ranking.
    p_auc = pr_auc(labels, probs)

    # Calibration.
    if true_solution_prob is not None:
        tprobs = np.asarray([float(true_solution_prob(c)) for c in cands])
        brier = float(np.mean((probs - tprobs) ** 2))
        ece = _ece(tprobs, probs)
    else:
        brier = float("nan")
        ece = float("nan")

    return RecoveryEvalResult(
        n_unqueried=len(cands),
        unseen_recall=recall,
        unseen_precision=precision,
        unseen_f1=f1,
        pr_auc=p_auc,
        region_recall=region_recall,
        uncovered_distance=uncovered_distance,
        brier=brier,
        ece=ece,
    )


def aurc(recovery_by_budget: Mapping[int, float]) -> float:
    """Area Under the Recovery-vs-Budget curve (unit-normalized trapezoid)."""
    pairs = sorted((int(b), float(r)) for b, r in recovery_by_budget.items())
    if len(pairs) < 2 or pairs[-1][0] <= pairs[0][0]:
        return 0.0
    bmin, bmax = pairs[0][0], pairs[-1][0]
    area = 0.0
    for (b0, r0), (b1, r1) in zip(pairs, pairs[1:]):
        if b1 <= b0:
            continue
        area += 0.5 * (r0 + r1) * (b1 - b0)
    return float(area / (bmax - bmin))
