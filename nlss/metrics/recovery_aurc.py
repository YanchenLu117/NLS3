"""V7 metrics — Recovery-AURC and the solution-space recovery family
(EXPERIMENT_PLAN §8.1).

Recovery-AURC is the construction track's *unified primary summary*: given the
posterior field's ranking of the recovered solution space against an
evaluator-hidden reference set of KNOWN true solutions, it integrates recovery
error (risk = 1 - precision over covered top objects) across coverage.  Better
recovery concentrates true solutions early -> lower AURC-risk, higher AUPRC.

``reference`` is the ground-truth solution id set (HypoSpace exposes exactly
enumerated admissible sets; a benchmark supplies it as the hidden evaluator).
The metric itself stays deterministic and outcome-blind w.r.t. the confidential
``scores`` it is called on.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _ranked(scores: Mapping[str, float]) -> list[str]:
    """Objects ranked by descending score, ties broken stably by id."""
    return sorted(scores, key=lambda o: (-float(scores[o]), o))


def recovery_curve(
    scores: Mapping[str, float],
    reference: Iterable[str],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Precision / recall / risk=1-precision at every coverage point.

    Returns (coverages, recalls, precisions, risks) — each of length n+1 with
    the (0,0/1) start point so callers can integrate cleanly.
    """
    ref = set(reference)
    obj_ids = _ranked(scores)
    n = len(obj_ids)
    if n == 0:
        return [0.0], [1.0], [1.0], [0.0]
    ref_n = len(ref)
    covers: list[float] = [0.0]
    recalls: list[float] = [0.0]
    precisions: list[float] = [1.0]
    risks: list[float] = [0.0]
    tp = 0
    for i, oid in enumerate(obj_ids, start=1):
        if oid in ref:
            tp += 1
        cov = i / n
        rec = tp / ref_n if ref_n else 1.0
        prec = tp / i
        covers.append(cov)
        recalls.append(rec)
        precisions.append(prec)
        risks.append(1.0 - prec)
    return covers, recalls, precisions, risks


def _trapz(y: Sequence[float], x: Sequence[float]) -> float:
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def recovery_aurc(
    scores: Mapping[str, float],
    reference: Iterable[str],
) -> dict[str, Any]:
    """Summary statistics for the Recovery-AURC construction metric.

    Returns a JSON-serializable dict:
      - au_recall_coverage : area under recall-vs-coverage (higher = better)
      - au_prc             : AUC-PR (higher = better)
      - aurc_risk          : area under risk-vs-coverage (lower = better);
                             this is the plan's unified primary, reported as-is
                             plus normalized (aurc_risk_norm).
      - recall_at_{0.25,0.5,0.75} for the paper's normalized-progress checkpoints.
    """
    covers, recalls, precisions, risks = recovery_curve(scores, reference)
    au_recall = _trapz(recalls, covers)
    au_prc = _trapz(precisions, covers)
    aurc = _trapz(risks, covers)
    out: dict[str, Any] = {
        "au_recall_coverage": round(au_recall, 6),
        "au_prc": round(au_prc, 6),
        "aurc_risk": round(aurc, 6),
        "aurc_risk_norm": round(aurc / max(covers[-1], 1e-9), 6),
        "n_reference": len(set(reference)),
        "n_objects": len(scores),
    }
    for q in (0.25, 0.5, 0.75):
        # nearest coverage point at or above q
        idx = next((i for i, c in enumerate(covers) if c >= q), len(covers) - 1)
        out[f"recall_at_{q:g}"] = round(recalls[idx], 6)
    return out


def from_state(state, reference: Iterable[str] | None = None, *, eta: float = 0.5):
    """Recovery-AURC directly from a SolutionSpaceState posterior field.

    ``state`` exposes ``objects`` and ``posterior`` (MarginalPosterior per id,
    with ``solution_probability``).  Reference is the known true-solution id set.
    Returns None when no posterior has been fitted.
    """
    if state is None or getattr(state, "posterior", None) is None:
        return None
    scores: dict[str, float] = {}
    for oid in state.objects:
        try:
            m = state.posterior.marginal(oid)
            sp = m.solution_probability
            scores[oid] = float(sp) if sp is not None else 0.0
        except Exception:
            scores[oid] = 0.0
    return recovery_aurc(scores, reference or ())
