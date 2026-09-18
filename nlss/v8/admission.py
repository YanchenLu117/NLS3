"""V8 — online admission (Detail §3.1) and §2.3 round interface carriers.

Admission: a proposed representation is admitted only when
``LogicValid(P̃_t; L_τ) = 1`` AND the actor-visible fidelity vector meets its
registered thresholds.  The online logic report states NINE items; the
actor-declared completeness status is marked ``UNVERIFIED`` during the
campaign; the hidden evaluator never sends feedback.  M0 runs a frozen
symbolic consequence prover over the public generator DSL; M1 checks only
submitted relations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

COMPLETENESS_UNVERIFIED = "UNVERIFIED"

LOGIC_REPORT_ITEMS: tuple[str, ...] = (
    "typing_structure",
    "executability",
    "task_constraints",
    "provenance",
    "outcome_leakage",
    "semantic_distinctions",
    "relation_cover_overlap_validity",
    "completeness_status",
    "provisional_level",
)


@dataclass(frozen=True)
class LogicReport:
    """The nine online logic-report items (Detail §3.1)."""

    items: Mapping[str, Any]

    def __post_init__(self) -> None:
        missing = [name for name in LOGIC_REPORT_ITEMS if name not in self.items]
        if missing:
            raise ValueError(f"logic report missing items: {', '.join(missing)}")

    @property
    def logic_valid(self) -> bool:
        """LogicValid = 1: every hard check passed (completeness is exempt — it
        is UNVERIFIED during the campaign by protocol)."""
        for name in LOGIC_REPORT_ITEMS:
            if name in ("completeness_status", "provisional_level"):
                continue
            value = self.items[name]
            if value is not True:
                return False
        return True

    @property
    def provisional_level(self) -> int:
        return int(self.items["provisional_level"])

    @property
    def completeness_status(self) -> str:
        return str(self.items["completeness_status"])


class PublicLogicChecker(ABC):
    """Actor-visible logic checker.  M0 = frozen symbolic consequence prover
    over the public generator DSL (equality/refinement consequences + proof
    certificates, no access to W^eval or outcomes).  M1 = consequence search
    disabled, submitted relations only."""

    checker_id: str
    mode: str  # "M0_prover" | "M1_declared_only"

    @abstractmethod
    def check(self, proposal: Mapping[str, Any], context: Mapping[str, Any]) -> LogicReport: ...


class DeclaredOnlyChecker(PublicLogicChecker):
    """M1: validates only declared relations/covers (no consequence search)."""

    def __init__(self, hard_checks: Mapping[str, Callable[[Mapping[str, Any]], bool]]) -> None:
        self.checker_id = "m1_declared_only"
        self.mode = "M1_declared_only"
        self._checks = dict(hard_checks)

    def check(self, proposal: Mapping[str, Any], context: Mapping[str, Any]) -> LogicReport:
        items: dict[str, Any] = {}
        for name, check in self._checks.items():
            items[name] = bool(check(proposal))
        items["completeness_status"] = COMPLETENESS_UNVERIFIED
        items["provisional_level"] = 1 if items.get("relation_cover_overlap_validity") else 0
        return LogicReport(items)


@dataclass(frozen=True)
class FidelityGate:
    """Registered probe thresholds: every registered probe must meet its bound."""

    thresholds: Mapping[str, float]

    def passed(self, fidelity_vector: Mapping[str, float]) -> bool:
        for probe, bound in self.thresholds.items():
            value = fidelity_vector.get(probe)
            if value is None or float(value) < float(bound):
                return False
        return True


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    logic_report: LogicReport
    fidelity_vector: Mapping[str, float]
    reasons: tuple[str, ...] = field(default_factory=tuple)


def admit(
    logic_report: LogicReport,
    fidelity_vector: Mapping[str, float],
    gate: FidelityGate,
) -> AdmissionDecision:
    """Online admission verdict (Detail §3.1): LogicValid AND fidelity gates."""
    reasons: list[str] = []
    if not logic_report.logic_valid:
        failed = [
            name
            for name in LOGIC_REPORT_ITEMS
            if name not in ("completeness_status", "provisional_level") and logic_report.items[name] is not True
        ]
        reasons.append(f"LogicValid=0: {', '.join(failed)}")
    if not gate.passed(fidelity_vector):
        missing = [
            probe
            for probe, bound in gate.thresholds.items()
            if fidelity_vector.get(probe) is None or float(fidelity_vector[probe]) < float(bound)
        ]
        reasons.append(f"fidelity below thresholds: {', '.join(missing)}")
    return AdmissionDecision(
        admitted=not reasons,
        logic_report=logic_report,
        fidelity_vector=dict(fidelity_vector),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# §2.3 per-round interface: six inputs, six outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskDossier:
    task_id: str
    public_description_ref: str
    validator_interface_ref: str
    initial_evidence_ids: tuple[str, ...]
    legality_rule_id: str


@dataclass(frozen=True)
class RemainingBudgets:
    oracle: int
    resource: Mapping[str, float]


@dataclass(frozen=True)
class StoppingRule:
    rule_id: str
    predicate_ref: str


@dataclass(frozen=True)
class RoundInputs:
    task_dossier: TaskDossier
    current_evidence: tuple[Mapping[str, Any], ...]
    legal_candidate_ids: tuple[str, ...]
    remaining_budgets: RemainingBudgets
    stopping_rule: StoppingRule


@dataclass(frozen=True)
class RecoveryPrediction:
    """Score for every legal candidate ID, or a benchmark-native submitted set.

    A missing prediction receives the P0-registered failed-checkpoint score
    (Detail §2.3) — carried by ``missing=True`` at the carrier level."""

    scores: Mapping[str, float] | None = None
    submitted_set: tuple[str, ...] | None = None
    missing: bool = False


@dataclass(frozen=True)
class RoundOutputs:
    proposed_candidate_ids: tuple[str, ...]
    persistent_state_ref: str
    recovery_prediction: RecoveryPrediction
    decision_trace_ref: str
    resource_usage: Mapping[str, float]
    completion_status: str
