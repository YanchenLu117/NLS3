"""NLSSLoop — the orchestration surface.  The loop does NOT implement an
acquisition policy: which objects are evaluated is decided by the scientific
agent / benchmark controller and passed in via ``selected_object_ids``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import (
    GroundedObject,
    LanguageHypothesis,
    Observation,
    RevisionGraph,
    ScientificGraph,
    VerificationResult,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..llm.base import LLMProvider
    from ..recovery.base import RecoveryBackend
    from .evidence import EvidenceBuilder
    from .interfaces import TaskAdapter
    from .state import SolutionSpaceState
    from ..state_strategies.base import StateStrategy


class NLSSLoop:
    def __init__(
        self,
        adapter: "TaskAdapter",
        llm: "LLMProvider",
        recovery: "RecoveryBackend",
        state_strategy: "StateStrategy",
        evidence_builder: "EvidenceBuilder",
    ) -> None:
        self.adapter = adapter
        self.llm = llm
        self.recovery = recovery
        self.state_strategy = state_strategy
        self.evidence_builder = evidence_builder

    async def propose(
        self,
        state: "SolutionSpaceState",
        task_prompt: str,
        n_proposals: int,
    ) -> tuple[LanguageHypothesis, ...]:
        raise NotImplementedError(
            "proposal generation is the scientific agent's responsibility; "
            "the loop only grounds/relates/recovers"
        )

    async def ground(
        self,
        hypotheses: tuple[LanguageHypothesis, ...],
    ) -> tuple[GroundedObject, ...]:
        """Compile -> Verify -> Canonicalize each hypothesis."""
        out: list[GroundedObject] = []
        for h in hypotheses:
            raw = await self.adapter.compile(h, self.llm)
            verdict = self.adapter.verify(raw)
            if not verdict.valid:
                if verdict.repaired_object is None:
                    continue
                raw = verdict.repaired_object
                verdict = self.adapter.verify(raw)
                if not verdict.valid:
                    continue
            obj = self.adapter.canonicalize(raw)
            out.append(obj)
        return tuple(out)

    def rebuild_relations(self, state: "SolutionSpaceState") -> None:
        objects = tuple(state.objects.values())
        state.scientific_graph = self.adapter.build_scientific_graph(objects)
        state.revision_graph = self.adapter.build_revision_graph(objects)

    def recover(
        self,
        state: "SolutionSpaceState",
        solution_threshold: float | None = None,
    ) -> None:
        from ..recovery.base import RecoveryInput

        # P0-1 / T3: forward the gamma (value-space) threshold — read back from
        # the facade's ``value_threshold_gamma`` metadata — so
        # ``marginal().solution_probability`` (the value walk p=norm_cdf) and
        # the S_t solution set are materialized, not left None.  eta (the
        # probability cutoff) is loop-agnostic here; it is consumed by the
        # facade's ``solution_set`` / readout family views, not by the backend.
        if solution_threshold is None:
            solution_threshold = float(
                state.metadata.get("value_threshold_gamma", state.metadata.get("solution_threshold", 0.0)) or 0.0
            )
        state.posterior = self.recovery.fit_posterior(
            RecoveryInput(
                objects=tuple(state.objects.values()),
                scientific_graph=state.scientific_graph,
                observations=tuple(state.observations.values()),
                solution_threshold=solution_threshold,
            )
        )

    async def evaluate_selected(
        self,
        state: "SolutionSpaceState",
        selected_object_ids: tuple[str, ...],
    ) -> tuple[Observation, ...]:
        observations: list[Observation] = []
        for oid in selected_object_ids:
            if oid not in state.objects:
                continue
            obs = await self.adapter.evaluate(state.objects[oid])
            observations.append(obs)
        return tuple(observations)
