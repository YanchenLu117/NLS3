"""V7 Core — evidence-driven representation revision (proposal §21-§25).

Two timescales:
  - fast:  new evidence only re-fits the field  B_t -> B_{t+1};
  - slow:  representation failure revises  (C_t, Gamma_t) -> (C_{t+1}, Gamma_{t+1}).

Four triggers (proposal §22):
  1. UNGROUNDABLE  — a proposed hypothesis lies outside dom(Gamma_t) and cannot
                     be grounded into the current representation.
  2. FIDELITY      — a scientifically distinct pair is collapsed: F_t < eps_F.
  3. EMPIRICAL     — persistent poor calibration / non-discriminative posterior
                     / failed region recovery (missing variable / relation).
  4. CONTRADICTION — new evidence persistently conflicts with structural
                     assumptions (latent context, category split, relation change).

Revision invariant:  D_{t+1} superset of D_t  (evidence intact; field rebuilt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TYPE_CHECKING

from .representation import (
    FidelityGate,
    LogicGate,
    LogicReport,
    RepresentationParadigm,
    RepresentationSpecification,
)

if TYPE_CHECKING:  # pragma: no cover
    from .interfaces import TaskAdapter
    from .state import SolutionSpaceState


class RevisionTrigger(str, Enum):
    UNGROUNDABLE = "ungroundable"
    FIDELITY = "fidelity"
    EMPIRICAL = "empirical"
    CONTRADICTION = "contradiction"


# Marginal EMPIRICAL tolerance: below this the field is non-discriminative.
_EMPIRICAL_MIN_DISCRIMINATION: float = 1e-9


@dataclass(frozen=True, slots=True)
class RevisionSignal:
    """A record of which triggers are currently firing (and why)."""

    fired: tuple[RevisionTrigger, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)

    def fires(self, trigger: RevisionTrigger) -> bool:
        return trigger in self.fired

    @property
    def any(self) -> bool:
        return len(self.fired) > 0

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(t.value for t in self.fired)


@dataclass(frozen=True, slots=True)
class RevisionReport:
    """Outcome of one revision decision."""

    needed: bool
    signals: RevisionSignal
    delta_description: str = ""
    new_spec: RepresentationSpecification | None = None
    adhered_logically: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def triggers(self) -> tuple[str, ...]:
        return self.signals.labels


class RepresentationRevisioner:
    """Scans persistent state for revision triggers and proposes a revised spec.

    No-LLM deterministic behavior: on any trigger, the revisioner returns a
    *tightened* spec (keeps only the semantic commitments the adapter can
    actually expose) plus a delta description naming the fired triggers.  With
    an LLM, ``propose_revision_spec`` would be the full LLM-Reinduce path
    (proposal §23); here it returns the deterministic tightening so the loop is
    testable without a gateway.
    """

    def __init__(
        self,
        adapter: "TaskAdapter",
        *,
        epsilon_f: float = 0.5,
        min_discrimination: float = _EMPIRICAL_MIN_DISCRIMINATION,
    ) -> None:
        self.adapter = adapter
        self.epsilon_f = epsilon_f
        self.min_discrimination = min_discrimination
        self.logic = LogicGate(adapter)
        self.fidelity = FidelityGate(adapter)

    # -- signal detection -------------------------------------------------

    def _signal_ungroundable(
        self,
        state: "SolutionSpaceState",
        spec: RepresentationSpecification,
    ) -> tuple[bool, str]:
        # Hypotheses for which grounding produced no object already left the
        # state; the facade records an ungroundable counter.  If the spec's
        # provenance reports one, or the adapter cannot render all objects,
        # we flag it.
        n_ungroundable = int(state.metadata.get("ungroundable_hypotheses", 0) or 0)
        if n_ungroundable > 0:
            return True, f"{n_ungroundable} hypothesis(ies) ungroundable in current representation"
        return False, ""

    def _signal_fidelity(
        self,
        state: "SolutionSpaceState",
        spec: RepresentationSpecification,
    ) -> tuple[bool, str]:
        f = self.fidelity.score(spec)
        if f < self.epsilon_f:
            return True, f"fidelity F_t={f:.3f} < epsilon_f={self.epsilon_f}"
        return False, ""

    def _signal_empirical(
        self,
        state: "SolutionSpaceState",
        spec: RepresentationSpecification,
    ) -> tuple[bool, str]:
        disc = bool(state.metadata.get("posterior_discriminative", False))
        if not disc:
            return True, "posterior non-discriminative (field provides no resolution)"
        return False, ""

    def _signal_contradiction(
        self,
        state: "SolutionSpaceState",
        spec: RepresentationSpecification,
    ) -> tuple[bool, str]:
        # Contradiction is structurally specific; here we honor an explicit,
        # evidence-backed flag (e.g. set by an LLM/analysis pass) rather than
        # inventing one from thin air.
        if state.metadata.get("representation_contradiction", False):
            return True, "evidence persistently conflicts with structural assumptions"
        return False, ""

    def scan(
        self,
        state: "SolutionSpaceState",
        spec: RepresentationSpecification,
        *,
        paradigm: RepresentationParadigm = RepresentationParadigm.V7_FULL,
    ) -> RevisionSignal:
        """Evaluate the four triggers.  V1/V6 (non-adaptive) never fire."""
        if not paradigm.is_adaptive:
            return RevisionSignal(fired=(), reasons={"paradigm": paradigm.value})
        fired: list[tuple[RevisionTrigger, str]] = []
        for trig, fn in (
            (RevisionTrigger.UNGROUNDABLE, self._signal_ungroundable),
            (RevisionTrigger.FIDELITY, self._signal_fidelity),
            (RevisionTrigger.EMPIRICAL, self._signal_empirical),
            (RevisionTrigger.CONTRADICTION, self._signal_contradiction),
        ):
            hit, reason = fn(state, spec)
            if hit:
                fired.append((trig, reason))
        return RevisionSignal(
            fired=tuple(t for t, _ in fired),
            reasons={str(t.value): r for t, r in fired},
        )

    # -- revision proposal ------------------------------------------------

    def propose_revision(
        self,
        spec: RepresentationSpecification,
        signals: RevisionSignal,
        *,
        paradigm: RepresentationParadigm = RepresentationParadigm.V7_FULL,
        llm_inductor=None,
    ) -> RepresentationSpecification:
        """Return a candidate revised spec.

        When ``llm_inductor`` is supplied and the trigger fires, it is given the
        chance to produce a STRUCTURAL revision (may add variables/relations via
        LLM-Revise, EXPERIMENT_PLAN §2.3).  If it yields a candidate it is used;
        otherwise the deterministic tightening path below runs.

        Deterministic path: tightens semantic_commitments to exactly what the
        adapter can expose (removing collapsed/unrealizable distinctions), the
        fidelity-repair direction.  It is the no-LLM fallback.
        """
        if not paradigm.is_adaptive or not signals.any:
            return spec
        if llm_inductor is not None:
            cand = llm_inductor(spec, signals)
            if cand is not None:
                return cand
        exposed = self.fidelity._exposed_dimensions()
        kept = tuple(c for c in spec.semantic_commitments if c in exposed)
        rev = spec.with_commitments(kept) if kept else spec.with_commitments(("structural_domain",))
        return RepresentationSpecification(
            spec_id=f"{spec.spec_id}.r",
            name=spec.name,
            description=spec.description + " [revised]",
            typed_variables=spec.typed_variables,
            relations=spec.relations,
            transforms=spec.transforms,
            semantic_commitments=rev.semantic_commitments,
            entities=spec.entities,          # keep Sigma (not reverted on revision)
            geometry=spec.geometry,          # keep K
            grounding=spec.grounding,        # keep G
            backend_hint=spec.backend_hint,
            provenance={**dict(spec.provenance), "revision_triggers": signals.labels},
            source="revision",
            admitted=spec.admitted,
        )

    # -- full decide+admit loop -------------------------------------------

    def decide_and_revise(
        self,
        state: "SolutionSpaceState",
        spec: RepresentationSpecification,
        *,
        paradigm: RepresentationParadigm = RepresentationParadigm.V7_FULL,
        epsilon_f: float | None = None,
        llm_inductor=None,
    ) -> RevisionReport:
        """Scan, and if needed propose a revision and re-admit it.

        Returns a :class:`RevisionReport`; the caller (facade) is responsible
        for adopting the ``new_spec`` into persistent state under the
        ``D_{t+1} superset D_t`` invariant.
        """
        signals = self.scan(state, spec, paradigm=paradigm)
        if not signals.any:
            return RevisionReport(needed=False, signals=signals)
        new_spec = self.propose_revision(spec, signals, paradigm=paradigm,
                                         llm_inductor=llm_inductor)
        eps = epsilon_f if epsilon_f is not None else self.epsilon_f
        # dual gate (logic + fidelity) on the revised candidate
        if paradigm.runs_logic_gate:
            report: LogicReport = self.logic.check(new_spec)
            logic_ok = report.valid
        else:
            logic_ok = True
        if paradigm.runs_fidelity_gate:
            fidelity_ok, _ = self.fidelity.admissible(new_spec, eps)
        else:
            fidelity_ok = True
        admitted = bool(logic_ok and fidelity_ok)
        new_spec = RepresentationSpecification(
            spec_id=new_spec.spec_id,
            name=new_spec.name,
            description=new_spec.description,
            typed_variables=new_spec.typed_variables,
            relations=new_spec.relations,
            transforms=new_spec.transforms,
            semantic_commitments=new_spec.semantic_commitments,
            entities=new_spec.entities,
            geometry=new_spec.geometry,
            grounding=new_spec.grounding,
            backend_hint=new_spec.backend_hint,
            provenance=new_spec.provenance,
            source=new_spec.source,
            admitted=admitted,
        )
        delta = "; ".join(signals.labels)
        return RevisionReport(
            needed=True,
            signals=signals,
            delta_description=delta,
            new_spec=new_spec,
            adhered_logically=admitted,
            metadata={"logic_valid": logic_ok, "fidelity_ok": fidelity_ok, "epsilon_f": eps},
        )
