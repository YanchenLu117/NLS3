"""V8 P0-c — governed revision routing (Detail §3.2).

Every revision event carries exactly one top-level trigger:

    GROUNDING (SEMANTIC_GAP | REALIZATION_GAP) | FIDELITY | EMPIRICAL |
    CONTRADICTION

P0 defines ONE deterministic predicate per trigger.  A trigger with no frozen
predicate CANNOT initiate a scored revision — the router only evaluates
registered predicates, and campaigns must pass :meth:`RevisionRouter.require_frozen`
for every trigger they intend to use.  Every revision stores the full required
record (pre-state, triggering evidence, diff, verdicts, acceptance, migration
status, re-grounding coverage, field rebuild, post metrics).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

MIGRATION_REENCODED = "REENCODED"
MIGRATION_UNREPRESENTABLE = "UNREPRESENTABLE"


class TriggerKind(str, Enum):
    GROUNDING = "GROUNDING"
    FIDELITY = "FIDELITY"
    EMPIRICAL = "EMPIRICAL"
    CONTRADICTION = "CONTRADICTION"


class GroundingSubkind(str, Enum):
    SEMANTIC_GAP = "SEMANTIC_GAP"
    REALIZATION_GAP = "REALIZATION_GAP"


# Deterministic routing precedence (order of Detail §3.2)
ROUTE_ORDER: tuple[TriggerKind, ...] = (
    TriggerKind.GROUNDING,
    TriggerKind.FIDELITY,
    TriggerKind.EMPIRICAL,
    TriggerKind.CONTRADICTION,
)


@dataclass(frozen=True)
class PredicateVerdict:
    triggered: bool
    detail: Mapping[str, Any] = field(default_factory=dict)


class FrozenPredicate(ABC):
    """One P0-frozen deterministic predicate for a revision trigger."""

    kind: TriggerKind
    subkind: GroundingSubkind | None = None
    predicate_id: str

    @abstractmethod
    def evaluate(self, context: Mapping[str, Any]) -> PredicateVerdict: ...


class ProbeThresholdPredicate(FrozenPredicate):
    """FIDELITY and both GROUNDING gap types: fires when a probe value breaches
    its registered threshold (direction: ``below`` or ``above``)."""

    def __init__(
        self,
        kind: TriggerKind,
        predicate_id: str,
        probe_key: str,
        threshold: float,
        *,
        direction: str = "below",
        subkind: GroundingSubkind | None = None,
    ) -> None:
        if kind not in (TriggerKind.FIDELITY, TriggerKind.GROUNDING):
            raise ValueError("ProbeThresholdPredicate covers FIDELITY and GROUNDING only")
        if direction not in ("below", "above"):
            raise ValueError("direction must be 'below' or 'above'")
        self.kind = kind
        self.subkind = subkind
        self.predicate_id = predicate_id
        self._probe_key = probe_key
        self._threshold = float(threshold)
        self._direction = direction

    def evaluate(self, context: Mapping[str, Any]) -> PredicateVerdict:
        probes = context.get("probe_values", {})
        value = probes.get(self._probe_key)
        if value is None:
            return PredicateVerdict(False, {"reason": "probe_absent", "probe_key": self._probe_key})
        value = float(value)
        breached = value < self._threshold if self._direction == "below" else value > self._threshold
        return PredicateVerdict(
            breached,
            {
                "probe_key": self._probe_key,
                "value": value,
                "threshold": self._threshold,
                "direction": self._direction,
            },
        )


class EmpiricalPredicate(FrozenPredicate):
    """EMPIRICAL: metric breaches threshold for N consecutive checkpoints."""

    def __init__(
        self,
        predicate_id: str,
        metric: str,
        threshold: float,
        consecutive_checkpoints: int,
        *,
        direction: str = "below",
    ) -> None:
        if consecutive_checkpoints < 1:
            raise ValueError("consecutive_checkpoints must be >= 1")
        if direction not in ("below", "above"):
            raise ValueError("direction must be 'below' or 'above'")
        self.kind = TriggerKind.EMPIRICAL
        self.predicate_id = predicate_id
        self._metric = metric
        self._threshold = float(threshold)
        self._k = int(consecutive_checkpoints)
        self._direction = direction

    def evaluate(self, context: Mapping[str, Any]) -> PredicateVerdict:
        history = context.get("checkpoint_history", [])
        window = history[-self._k :]
        if len(window) < self._k:
            return PredicateVerdict(False, {"reason": "insufficient_history", "have": len(window), "need": self._k})
        for point in window:
            value = point.get(self._metric)
            if value is None:
                return PredicateVerdict(False, {"reason": "metric_absent", "metric": self._metric})
            value = float(value)
            breached = value < self._threshold if self._direction == "below" else value > self._threshold
            if not breached:
                return PredicateVerdict(
                    False,
                    {"reason": "streak_broken", "metric": self._metric, "value": value},
                )
        return PredicateVerdict(
            True,
            {
                "metric": self._metric,
                "threshold": self._threshold,
                "consecutive_checkpoints": self._k,
                "direction": self._direction,
            },
        )


class ContradictionPredicate(FrozenPredicate):
    """CONTRADICTION: a machine-checkable invariant is violated.

    ``invariant(state)`` returns a falsy value when the invariant holds, or a
    truthy violation description.  Blinded-rubric contradictions are handled by
    the campaign layer, not by a runtime predicate.
    """

    def __init__(self, predicate_id: str, invariant: Callable[[Mapping[str, Any]], Any], invariant_id: str) -> None:
        self.kind = TriggerKind.CONTRADICTION
        self.predicate_id = predicate_id
        self._invariant = invariant
        self._invariant_id = invariant_id

    def evaluate(self, context: Mapping[str, Any]) -> PredicateVerdict:
        violation = self._invariant(context.get("invariant_state", {}))
        if not violation:
            return PredicateVerdict(False, {"invariant": self._invariant_id})
        return PredicateVerdict(True, {"invariant": self._invariant_id, "violation": violation})


@dataclass
class RevisionEvent:
    """Full required record for one revision event (Detail §3.2)."""

    trigger: Mapping[str, Any]
    triggering_evidence: Mapping[str, Any]
    pre_state_ref: str
    proposed_diff: Mapping[str, Any]
    logic_verdict: Mapping[str, Any]
    fidelity_verdict: Mapping[str, Any]
    accepted: bool | None = None
    migration_status: str | None = None
    regrounding_coverage: Mapping[str, Any] | None = None
    field_rebuild_ref: str | None = None
    post_metrics: Mapping[str, Any] | None = None
    event_id: str = ""

    REQUIRED = (
        "trigger",
        "triggering_evidence",
        "pre_state_ref",
        "proposed_diff",
        "logic_verdict",
        "fidelity_verdict",
        "accepted",
        "migration_status",
        "regrounding_coverage",
        "field_rebuild_ref",
        "post_metrics",
    )

    def is_complete(self) -> list[str]:
        missing = [name for name in self.REQUIRED if getattr(self, name) is None]
        return missing

    def mark_accepted(self) -> None:
        """Acceptance gate (Detail §3.2): a revision may be accepted only when
        both the logic verdict and the fidelity verdict passed."""
        if not self.logic_verdict.get("logic_valid"):
            raise ValueError("cannot accept: logic verdict is not LogicValid=1")
        if not self.fidelity_verdict.get("passed"):
            raise ValueError("cannot accept: fidelity verdict below registered thresholds")
        self.accepted = True

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "trigger": copy.deepcopy(dict(self.trigger)),
            "triggering_evidence": copy.deepcopy(dict(self.triggering_evidence)),
            "pre_state_ref": self.pre_state_ref,
            "proposed_diff": copy.deepcopy(dict(self.proposed_diff)),
            "logic_verdict": copy.deepcopy(dict(self.logic_verdict)),
            "fidelity_verdict": copy.deepcopy(dict(self.fidelity_verdict)),
            "accepted": self.accepted,
            "migration_status": self.migration_status,
            "regrounding_coverage": None if self.regrounding_coverage is None else copy.deepcopy(dict(self.regrounding_coverage)),
            "field_rebuild_ref": self.field_rebuild_ref,
            "post_metrics": None if self.post_metrics is None else copy.deepcopy(dict(self.post_metrics)),
        }
        return payload


class UnfrozenTriggerError(RuntimeError):
    """A campaign planned to use a trigger kind that has no frozen predicate."""


@dataclass
class MigrationReport:
    """Evidence migration under a new contract (Detail §3.2, final paragraph)."""

    reencoded: list[dict[str, Any]]
    unrepresentable: list[dict[str, Any]]
    from_version: Any
    to_version: Any

    @property
    def excluded_count(self) -> int:
        return len(self.unrepresentable_ids)

    @property
    def unrepresentable_ids(self) -> list[str]:
        return [str(r.get("exp_id", i)) for i, r in enumerate(self.unrepresentable)]


def migrate_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    from_version: Any,
    to_version: Any,
    encoder: Callable[[Mapping[str, Any], Any], dict[str, Any] | None],
) -> MigrationReport:
    """Re-encode every historical experiment under the new contract.

    ``encoder(record, new_contract_version)`` returns the re-encoded record, or
    None when the record is UNREPRESENTABLE under the new contract — unencodable
    records are KEPT (never discarded or fabricated) and counted; the attached
    field is rebuilt from the successfully re-encoded observations only."""
    reencoded: list[dict[str, Any]] = []
    unrepresentable: list[dict[str, Any]] = []
    for record in records:
        try:
            encoded = encoder(record, to_version)
        except Exception as exc:  # encoder bugs must not silently drop evidence
            encoded = None
            record = {**record, "migration_error": str(exc)[:200]}
        if encoded is None:
            unrepresentable.append({**dict(record), "migration_status": MIGRATION_UNREPRESENTABLE})
        else:
            reencoded.append({**dict(encoded), "migration_status": MIGRATION_REENCODED})
    return MigrationReport(
        reencoded=reencoded,
        unrepresentable=unrepresentable,
        from_version=from_version,
        to_version=to_version,
    )


class RevisionRouter:
    """Evaluates registered (frozen) predicates in Detail §3.2 order."""

    def __init__(self) -> None:
        self._predicates: dict[TriggerKind, list[FrozenPredicate]] = {kind: [] for kind in TriggerKind}

    def register(self, predicate: FrozenPredicate) -> None:
        self._predicates[predicate.kind].append(predicate)

    def require_frozen(self, kinds: tuple[TriggerKind, ...] | list[TriggerKind]) -> None:
        """Assert every trigger kind the campaign intends to use has a frozen
        predicate (P0 discipline; UnfrozenTriggerError otherwise)."""
        missing = [kind.value for kind in kinds if not self._predicates[kind]]
        if missing:
            raise UnfrozenTriggerError(
                f"no frozen predicate registered for: {', '.join(missing)} — cannot initiate scored revisions"
            )

    def route(
        self, context: Mapping[str, Any]
    ) -> tuple[TriggerKind | None, FrozenPredicate | None, PredicateVerdict | None]:
        """Return (kind, predicate, verdict) of the first triggered frozen
        predicate, or (None, None, None).  Only registered predicates are
        evaluated: a trigger kind with no frozen predicate can never fire."""
        for kind in ROUTE_ORDER:
            for predicate in self._predicates[kind]:
                verdict = predicate.evaluate(context)
                if verdict.triggered:
                    return kind, predicate, verdict
        return None, None, None

    def make_event(
        self,
        kind: TriggerKind,
        predicate: FrozenPredicate,
        verdict: PredicateVerdict,
        *,
        triggering_evidence: Mapping[str, Any],
        pre_state_ref: str,
        proposed_diff: Mapping[str, Any],
        logic_verdict: Mapping[str, Any],
        fidelity_verdict: Mapping[str, Any],
        event_id: str = "",
    ) -> RevisionEvent:
        trigger: dict[str, Any] = {
            "kind": kind.value,
            "predicate_id": predicate.predicate_id,
            "detail": dict(verdict.detail),
        }
        if kind is TriggerKind.GROUNDING:
            subkind = predicate.subkind
            trigger["subkind"] = subkind.value if subkind else None
        return RevisionEvent(
            trigger=trigger,
            triggering_evidence=copy.deepcopy(dict(triggering_evidence)),
            pre_state_ref=pre_state_ref,
            proposed_diff=copy.deepcopy(dict(proposed_diff)),
            logic_verdict=copy.deepcopy(dict(logic_verdict)),
            fidelity_verdict=copy.deepcopy(dict(fidelity_verdict)),
            event_id=event_id,
        )
