"""V8 P0-f — common recovery and curve metrics (Detail §6).

For each paired comparison at checkpoint budget b_j there is ONE common unseen
evaluation universe U_j^common = U^legal \\ (D_0 ∪ ⋃_m Q_{m,≤j}) — every
compared arm is scored on the same universe.  For finite ranked outputs R_j is
average precision on U_j^common for membership in S*_j (threshold-free: a
predict-everything strategy cannot score perfect recovery).  Ties in evaluator
thresholds break by the frozen candidate-ID order until the preregistered
prevalence is reached.  If S*_j = ∅ or U_j^common = ∅, EVERY arm receives
R_j = 1 under the contrast-neutral convention and the checkpoint is marked
``collectively_depleted``; a sensitivity AURC truncated at the preceding
checkpoint is reported.  No arm-specific replacement universe is allowed.

Recovery-AURC = Σ (R_{j-1}+R_j)/2 · (u_j − u_{j-1}) with u_j the normalized
budget; utility curves use the same trapezoidal scalarization (utility-AUC).
Method failure: R_j = 0 and utility = the P0 worst legal value from the failure
checkpoint onward (Detail §6 / §14 — failures stay in the denominator).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

COLLECTIVELY_DEPLETED = "collectively_depleted"


class MetricsProtocolError(ValueError):
    """Metric inputs violate the §6 contract."""


def common_universe(
    legal_ids: set[str],
    initial_ids: set[str],
    queried_by_arm: Mapping[str, set[str]],
) -> set[str]:
    """U_j^common = U^legal \\ (D_0 ∪ ⋃_m Q_{m,≤j}) — identical for all arms."""
    queried_union: set[str] = set()
    for queried in queried_by_arm.values():
        queried_union |= queried
    return legal_ids - initial_ids - queried_union


def average_precision(
    scores: Mapping[str, float],
    solution_ids: set[str],
    universe: set[str],
) -> float:
    """Average precision on the common universe (Detail §6: BH/GB1 R_j).

    Rank by score descending, ties by frozen candidate-ID ascending; precision
    averaged at each positive hit.  Predict-everything cannot game this: the
    denominator is the number of positives in the universe, and low-ranked
    positives contribute proportionally to their rank.
    """
    universe_items = [(cid, scores[cid]) for cid in sorted(universe) if cid in scores]
    if not universe_items:
        raise MetricsProtocolError("common universe is empty (use the depleted convention upstream)")
    ranked = sorted(universe_items, key=lambda kv: (-kv[1], kv[0]))
    hits = 0
    total_precision = 0.0
    for rank, (cid, _score) in enumerate(ranked, start=1):
        if cid in solution_ids:
            hits += 1
            total_precision += hits / rank
    positives = sum(1 for cid in universe if cid in solution_ids)
    if positives == 0:
        raise MetricsProtocolError("no positives in the common universe (use the depleted convention)")
    return total_precision / positives


def checkpoint_recovery(
    *,
    universe: set[str],
    solution_ids: set[str],
    scores: Mapping[str, float] | None = None,
    submitted_set: set[str] | None = None,
) -> tuple[float, bool]:
    """R_j at one checkpoint under the §6 conventions.

    Either a score for every legal candidate ID (BH/GB1: average precision on
    the common universe) or a benchmark-native submitted set (HypoSpace: official
    recovery = |S* ∩ submitted| / |S*| computed on the common universe).  If the
    common universe is empty or contains no solution members, EVERY arm receives
    R_j = 1 and the checkpoint is collectively depleted (contrast-neutral).
    """
    if not universe:
        return 1.0, True
    depleted_solution = not (universe & solution_ids)
    if depleted_solution:
        return 1.0, True
    if submitted_set is not None:
        positives = universe & solution_ids
        recovered = positives & set(submitted_set)
        return len(recovered) / len(positives), False
    if scores is None:
        raise MetricsProtocolError("either scores or a submitted set is required")
    return average_precision(scores, solution_ids, universe), False


def recovery_aurc(
    budgets: Sequence[int],
    recoveries: Sequence[float],
) -> float:
    """Trapezoidal Recovery-AURC over normalized budget (Detail §6).

    u_j = (b_j − b_0) / (b_J − b_0); AURC = Σ (R_{j-1}+R_j)/2 · (u_j−u_{j-1}).
    """
    if len(budgets) != len(recoveries):
        raise MetricsProtocolError("budgets and recoveries must align per checkpoint")
    if len(budgets) < 2:
        raise MetricsProtocolError("at least two checkpoints are required")
    if budgets[0] != min(budgets) or budgets[-1] != max(budgets):
        raise MetricsProtocolError("budgets must be ascending checkpoints")
    span = budgets[-1] - budgets[0]
    if span <= 0:
        raise MetricsProtocolError("budget span must be positive")
    u = [(b - budgets[0]) / span for b in budgets]
    area = 0.0
    for j in range(1, len(budgets)):
        area += (recoveries[j - 1] + recoveries[j]) / 2.0 * (u[j] - u[j - 1])
    return area


def truncated_recovery_aurc(budgets: Sequence[int], recoveries: Sequence[float], *, upto_index: int) -> float:
    """Sensitivity AURC truncated at the checkpoint preceding a depleted one."""
    if len(budgets) < 2 or upto_index < 1:
        raise MetricsProtocolError("truncation needs at least two checkpoints before the depleted one")
    return recovery_aurc(budgets[:upto_index], recoveries[:upto_index])


def apply_failure_score(
    recoveries: Sequence[float],
    utilities: Sequence[float],
    *,
    failure_checkpoint_index: int,
    worst_legal_utility: float,
) -> tuple[list[float], list[float]]:
    """§6/§14 failure rule: from the failure checkpoint onward R_j = 0 and
    utility = the P0 worst legal value.  Failures stay in the denominator."""
    if failure_checkpoint_index < 0 or failure_checkpoint_index >= len(recoveries):
        raise MetricsProtocolError("failure checkpoint out of range")
    rec = [0.0 if i >= failure_checkpoint_index else float(r) for i, r in enumerate(recoveries)]
    util = [
        float(worst_legal_utility) if i >= failure_checkpoint_index else float(u)
        for i, u in enumerate(utilities)
    ]
    return rec, util


def utility_auc(budgets: Sequence[int], utilities: Sequence[float]) -> float:
    """Same normalized trapezoidal scalarization applied to a utility curve."""
    return recovery_aurc(budgets, utilities)
