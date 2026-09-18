"""V7 Core — Algorithm 1: the coupled fast+slow persistent-loop driver.

This module is additive.  It turns the three-operation facade
(:class:`~nlss.core.model.NLSSModel`) plus the V7 representation layer into a
real agent loop that alternates the two timescales of §04/method.tex:

  * **fast loop** — propose -> implement/ground -> evaluate -> re-fit the field
    ``B_t -> B_{t+1}``; evidence only re-fits the empirical field.
  * **slow loop** — on a representation-failure trigger, revise
    ``(C_t, Gamma_t) -> (C_{t+1}, Gamma_{t+1})`` by re-admitting a revised
    ``P_t`` through the dual gate; evidence ``D_t`` is never erased
    (``D_{t+1} superset D_t``).

The loop is the *orchestrator*: it never calls ``NLSSLoop.propose`` (which is
``NotImplementedError``).  Proposal authority is injected as an async
``proposer``; grounding / evaluation are injected as an ``implementer`` and an
``evaluator`` (with model-backed defaults provided here).  All primitive
methods called on the facade are the existing public API.

Per round (Algorithm 1):

    1. Build a language readout from ``(P_t, C_t, Gamma_t, B_t)`` via
       ``model.readout(...)`` + ``model.interpret_solution_space(...)``.
    2. Allocate ``SOL/BND/COV/OPEN`` proposal counts via
       ``model.batch_plan(batch_size)`` (threading the rounding residual into a
       ``slot_residual`` map kept on the loop).
    3. For each channel: ``SOL/BND/COV`` pick an in-space target through the
       same acquisition policy private to ``model.acquire`` (the exploration
       controller's normalized/rank policy), then ask the proposer for a
       hypothesis toward that target; ``OPEN`` calls ``model.open_directive()``
       and asks the proposer for an open-world hypothesis.
    4. Implement (ground/verify/canonicalize) and evaluate, appending to the
       evidence ledger via ``model.update``.
    5. ``model.recover()`` to re-fit ``B_t``.
    6. ``model.check_revision()``; if a representation-failure trigger fires,
       ``model.revise_representation()`` (which double-gates + refits).  If the
       revised spec was NOT admitted, the loop keeps the old spec and logs it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

from .acquisition import CHANNELS
from .model import EvidenceRecord, NLSSModel

if TYPE_CHECKING:  # pragma: no cover
    from .state import SolutionSpaceState
    from .types import GroundedObject, LanguageHypothesis, Observation


# ---------------------------------------------------------------------------
# Default implementer / evaluator wiring (model-backed).
#
# The loop ships with these so a caller only has to inject proposer logic; the
# implementer grounds hypotheses through the facade (``model.update`` with
# hypothesis records) and the evaluator evaluates objects through the facade
# (``model.update`` with observation records).
# ---------------------------------------------------------------------------


def default_implementer(model: NLSSModel):
    """Build an async ``implementer(hypotheses) -> tuple[GroundedObject, ...]``
    from a facade: it appends hypothesis records and returns the objects that
    were grounded for those hypotheses' ids."""

    async def _impl(hypotheses: Sequence["LanguageHypothesis"]):
        records = [
            EvidenceRecord(kind="hypothesis", hypothesis=h) for h in hypotheses
        ]
        model.update(records)
        wanted = {h.hypothesis_id for h in hypotheses}
        return tuple(
            o
            for o in model.state.objects.values()
            if wanted.intersection(o.source_hypothesis_ids)
        )

    return _impl


def default_evaluator(model: NLSSModel):
    """Build an async ``evaluator(obj) -> Observation | None`` from a facade: it
    evaluates through the adapter and appends an observation record."""

    async def _ev(obj: "GroundedObject") -> "Observation | None":
        obs = await model.adapter.evaluate(obj)
        model.update([EvidenceRecord(kind="observation", observation=obs)])
        return obs

    return _ev


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class NLSSAgentLoop:
    """Algorithm 1 — coupled fast+slow persistent-loop driver.

    Parameters
    ----------
    model:
        The :class:`NLSSModel` facade whose primitive methods the loop drives.
    proposer:
        Async ``proposer(channel, readout_text, task_prompt, n) ->
        Sequence[LanguageHypothesis]`` — the stand-in for the LLM proposer.
        The loop calls it once per channel with the per-slot ``n``.
    implementer:
        Async ``implementer(hypotheses) -> Sequence[GroundedObject]``.  Defaults
        to :func:`default_implementer` (grounds via ``model.update``).
    evaluator:
        Async ``evaluator(obj) -> Observation | None``.  Defaults to
        :func:`default_evaluator` (evaluates via ``model.update``).
    rng:
        Optional seeded RNG for reproducible channel sampling; a module-level
        ``random.Random()`` is used when None.
    """

    def __init__(
        self,
        model: NLSSModel,
        proposer,
        implementer=None,
        evaluator=None,
        *,
        rng: "random.Random | None" = None,
    ) -> None:
        self.model = model
        self.proposer = proposer
        self.implementer = implementer if implementer is not None else default_implementer(model)
        self.evaluator = evaluator if evaluator is not None else default_evaluator(model)
        self.rng = rng if rng is not None else random.Random()
        # carried batch_plan rounding residual (step 2) — converges channel
        # allocations to the long-run gear proportions across rounds.
        self._slot_residual: tuple[float, float, float, float] | None = None
        self._round: int = 0
        self._candidate_pool: Sequence[str] | None = None
        # audit trail of revision decisions + per-round summaries.
        self._log: list[dict[str, Any]] = []
        self.summaries: list[dict[str, Any]] = []

    # -- readout composition ---------------------------------------------

    def _build_readout(self, task_prompt: str) -> str:
        """Compose a language readout over (P_t, C_t, Gamma_t, B_t)."""
        model = self.model
        parts = [f"# Task: {task_prompt}"]
        # P_t + C_t + Gamma_t summary
        try:
            rep = model.representational_state()
            parts.append(
                f"## Representation P_t: {rep['P_t']['spec_id']} "
                f"(admitted={rep['P_t']['admitted']}, "
                f"commitments={rep['P_t']['semantic_commitments']})"
            )
            parts.append(f"## Empirical field C_t: backend={rep['C_t']['backend']}")
        except Exception as exc:  # pragma: no cover - defensive
            parts.append(f"## Representation summary unavailable: {exc}")
        # B_t computational-space readout
        try:
            comp = model.readout(model.state, token_budget=1200)
            parts.append("## Computational space (B_t / readout):\n" + comp)
        except Exception as exc:  # pragma: no cover - defensive
            parts.append(f"## Computational space readout unavailable: {exc}")
        # Gamma^left interpretation
        nl = model.interpret_solution_space(model.state)
        parts.append("## Interpreted solution space (Gamma^left):\n" + nl)
        if self._candidate_pool:
            parts.append(f"## Candidate pool available: {len(self._candidate_pool)} items")
        return "\n".join(parts)

    def _channel_context(self, channel: str, targets: Sequence[str], base: str) -> str:
        """Augment the readout with channel-specific acquisition guidance."""
        model = self.model
        lines = [base]
        if channel == "OPEN":
            directive = model.open_directive()
            lines.append(f"## {channel} directive\n{directive}")
            return "\n".join(lines)
        if targets:
            described = [
                model.adapter.render_object(model.state.objects[t]) for t in targets
            ]
            lines.append(f"## {channel} acquisition targets\n" + "; ".join(described))
        return "\n".join(lines)

    # -- within-space target selection -----------------------------------

    def _pick_targets(self, channel: str, n: int) -> list[str]:
        """Pick up to ``n`` in-space targets for SOL/BND/COV.

        Reuses the exact acquisition policy that ``model.acquire`` private
        (the exploration controller's masked-softmax ranking), but pinned to a
        single channel so the per-channel budget from ``batch_plan`` is
        respected (calling ``model.acquire`` would re-sample the channel and
        break the allocated counts).
        """
        model = self.model
        oids = list(model.state.objects)
        if not oids:
            return []
        ranked = model.exploration_controller.rank(
            channel, oids, model.state.posterior, model.eta
        )
        return list(ranked[: min(n, len(ranked))])

    # -- the Algorithm 1 round -------------------------------------------

    async def run_round(
        self,
        task_prompt: str,
        batch_size: int,
        candidate_pool: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Run one coupled fast+slow loop round.  Returns the per-round summary.

        Summary keys: ``round``, ``n_new_objects``, ``n_new_observations``,
        ``channels_used``, ``triggers``, ``representation_admitted``,
        ``field_rebuilt``, ``evidence_count`` (plus ``revision_fired``).
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
        model = self.model
        round_idx = self._round
        self._candidate_pool = list(candidate_pool) if candidate_pool else None

        # 1) fit B_t from current evidence, then build the language readout.
        state = model.recover()
        readout_text = self._build_readout(task_prompt)

        # 2) allocate per-channel counts via batch_plan (thread residual).
        counts, self._slot_residual = model.batch_plan(
            batch_size, self._slot_residual
        )

        # 3) propose per channel.
        proposed: dict[str, list] = {}
        channels_used: list[str] = []
        for ch in CHANNELS:
            n = int(counts.get(ch, 0) or 0)
            if n <= 0:
                continue
            if ch == "OPEN":
                ctx = self._channel_context(ch, [], readout_text)
                n_prop = n
            else:
                targets = self._pick_targets(ch, n)
                if not targets:  # no in-space candidates -> channel idle this round
                    continue
                ctx = self._channel_context(ch, targets, readout_text)
                n_prop = len(targets)
            hyps = await self.proposer(ch, ctx, task_prompt, n_prop)
            hyps = list(hyps or [])
            if hyps:
                proposed[ch] = hyps
                channels_used.append(ch)

        # 4) implement + ground, then evaluate -> append to the evidence ledger.
        n0_obj = len(model.state.objects)
        n0_obs = len(model.state.observations)
        for ch, hyps in proposed.items():
            grounded = await self.implementer(hyps) or ()
            for obj in grounded:
                await self.evaluator(obj)
        n_new_objects = len(model.state.objects) - n0_obj
        n_new_observations = len(model.state.observations) - n0_obs

        # 5) re-fit B_t.
        model.recover()

        # 6) representation-failure (slow) loop.
        triggers: list[str] = []
        revision_fired = False
        representation_admitted = bool(model.representation.admitted)
        field_rebuilt = False
        signal = model.check_revision()
        if signal.any:
            triggers = list(signal.labels)
            revision_fired = True
            old_spec = model.representation
            rev = await model.arevise_representation()  # async: re-grounding actually runs
            new_spec = rev.new_spec
            if rev.needed and new_spec is not None and not new_spec.admitted:
                # revised spec rejected by the dual gate: keep the old spec.
                model._representation = old_spec
                model.state.metadata["representation_field_rebuilt"] = False
                representation_admitted = False
                field_rebuilt = False
                self._log.append(
                    {
                        "round": round_idx,
                        "revision_not_admitted": True,
                        "proposed_spec_id": new_spec.spec_id,
                        "old_spec_id": old_spec.spec_id,
                        "triggers": triggers,
                    }
                )
            elif rev.needed:
                rep_admitted = (
                    new_spec.admitted if new_spec is not None else bool(old_spec.admitted)
                )
                representation_admitted = bool(rep_admitted)
                field_rebuilt = bool(
                    model.state.metadata.get("representation_field_rebuilt", False)
                )
                self._log.append(
                    {
                        "round": round_idx,
                        "revision_admitted": representation_admitted,
                        "spec_id": new_spec.spec_id if new_spec else None,
                        "triggers": triggers,
                    }
                )
            else:  # signal fired but no revision proposed (defensive)
                representation_admitted = bool(model.representation.admitted)

        summary = {
            "round": round_idx,
            "n_new_objects": n_new_objects,
            "n_new_observations": n_new_observations,
            "channels_used": channels_used,
            "triggers": triggers,
            "revision_fired": revision_fired,
            "representation_admitted": representation_admitted,
            "field_rebuilt": field_rebuilt,
            "evidence_count": len(model.evidence_log),
        }
        self.summaries.append(summary)
        model.state.round_index = round_idx + 1
        self._round += 1
        return summary

    # -- introspection ---------------------------------------------------

    def revision_log(self) -> tuple[dict[str, Any], ...]:
        """Audit trail of every revision decision made by the loop so far."""
        return tuple(self._log)


__all__ = ["NLSSAgentLoop", "default_implementer", "default_evaluator"]
