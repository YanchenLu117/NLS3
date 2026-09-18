"""V8 P0-d — formal intervention harnesses (Detail §4.3/§4.4).

Hidden-relation intervention (M0 vs M1): construct a true semantic relation,
remove it from the acting proposal while holding oracle evidence fixed, permit
an inconsistent grounding; FCR = #invalid representations admitted / #invalid
presented.  The frozen generator produces at least one invalid case per
evaluated unit (denominator nonzero by construction); a method-attributable
failure that prevents presentation receives FCR=1 for that unit (Detail §4.3).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .masking import EvaluatorUniverse


@dataclass(frozen=True)
class HiddenRelationCase:
    """One frozen intervention case (Detail §4.3)."""

    case_id: str
    true_relation: tuple[int, int]  # (a, b): a ⊑ b holds in W^eval
    removed_from_proposal: bool  # the relation was omitted by the actor
    inconsistent_grounding: bool  # g(A)=P, g(B)=Q with P⊄Q permitted
    presented: bool  # the invalid representation reached admission
    admitted: bool  # the online admission verdict
    method_attributable_failure: bool = False  # prevented presentation → FCR=1


def validate_hidden_relation_case(
    case: HiddenRelationCase,
    universe: EvaluatorUniverse,
    predicates: Mapping[int, Callable[[str], bool]],
) -> list[str]:
    """Pre-flight check that a case is well-formed per Detail §4.3."""
    problems: list[str] = []
    a, b = case.true_relation
    if a not in predicates or b not in predicates:
        problems.append(f"case {case.case_id}: predicate for generator {a}/{b} missing")
        return problems
    mask_a = universe.sigma(predicates[a])
    mask_b = universe.sigma(predicates[b])
    if mask_a & mask_b != mask_a:
        problems.append(f"case {case.case_id}: claimed true relation {a}⊑{b} does NOT hold in W^eval")
    if case.inconsistent_grounding and not case.removed_from_proposal:
        problems.append(f"case {case.case_id}: inconsistent grounding requires the relation to be removed")
    if case.admitted and not case.presented:
        problems.append(f"case {case.case_id}: admitted but not presented")
    return problems


@dataclass
class FCRScorer:
    """False Certification Rate over a family of intervention cases (§4.3)."""

    cases: list[HiddenRelationCase] = field(default_factory=list)

    def add(self, case: HiddenRelationCase) -> None:
        self.cases.append(case)

    def score(self) -> tuple[float, int, int]:
        """Returns (fcr, invalid_admitted, invalid_presented).

        Denominator = cases presented as invalid (presented with an inconsistent
        grounding).  A method-attributable failure that prevented presentation
        counts as an admitted invalid with worst-case weight (FCR=1 for the
        unit) — implemented by counting it in both numerator and denominator.
        """
        if not self.cases:
            raise ValueError("no intervention cases scored (frozen generator guarantees >=1 per unit)")
        numerator = 0
        denominator = 0
        for case in self.cases:
            if case.method_attributable_failure:
                numerator += 1
                denominator += 1
                continue
            if case.presented and case.inconsistent_grounding and case.removed_from_proposal:
                denominator += 1
                if case.admitted:
                    numerator += 1
        return numerator / denominator, numerator, denominator


# ------------------------------------------------------------------ §4.4


@dataclass(frozen=True)
class InterventionContrast:
    """The four preregistered formal ablation contrasts (Detail §4.4)."""

    contrast_id: str  # "M0_vs_M1" | "M0_vs_M2" | "M0_vs_M3" | "M0_vs_M4"
    frozen_intervention: str
    primary_endpoint: str


PREREGISTERED_CONTRASTS: tuple[InterventionContrast, ...] = (
    InterventionContrast("M0_vs_M1", "omitted true relation with inconsistent grounding", "FCR"),
    InterventionContrast("M0_vs_M2", "registered contrast/round-trip corruption", "semantic_round_trip_failure_rate"),
    InterventionContrast("M0_vs_M3", "remove backward interpretation on held-out state probes", "held_out_state_query_accuracy"),
    InterventionContrast("M0_vs_M4", "registered representation-failure trigger", "post_trigger_recovery_aurc"),
)


def semantic_round_trip_failure_rate(
    probe_results: Sequence[tuple[str, bool]],
    *,
    equality_tolerance: float = 0.0,
) -> tuple[float, int, int]:
    """Fraction of registered probe predicates failing the frozen
    equality/tolerance rule (Detail §4.4, M0 vs M2 primary endpoint).

    ``probe_results``: (probe_id, round_trip_passed).  Tolerance semantics are
    frozen at P0; the scorer counts a probe as failed when ``passed`` is False.
    """
    if not probe_results:
        raise ValueError("no registered probes (P0 guarantees a nonzero probe set)")
    failed = sum(1 for _, ok in probe_results if not ok)
    _ = equality_tolerance  # tolerance is frozen at P0 and baked into probe verdicts
    return failed / len(probe_results), failed, len(probe_results)


def held_out_state_query_accuracy(
    query_results: Sequence[tuple[str, str, str]],
    *,
    abstention_counts_as: str = "registered_rubric",
) -> tuple[float, int, int]:
    """Macro accuracy over frozen query classes (Detail §4.4, M0 vs M3 endpoint).

    ``query_results``: (query_class, predicted, canonical).  Abstention scoring
    follows the registered rubric; the default counts a wrong-class abstention
    as incorrect and a correct abstention as correct (both encoded by the
    caller as predicted == canonical or not).
    """
    _ = abstention_counts_as
    if not query_results:
        raise ValueError("no held-out state queries (P0 freezes a nonempty query set)")
    by_class: dict[str, list[bool]] = {}
    for q_class, predicted, canonical in query_results:
        by_class.setdefault(q_class, []).append(predicted == canonical)
    macro = sum(sum(v) / len(v) for v in by_class.values()) / len(by_class)
    correct = sum(sum(v) for v in by_class.values())
    return macro, correct, len(query_results)
