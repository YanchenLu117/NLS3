"""V7 Core — re-grounding and missing-data semantics on representation revision.

This module is additive.  It is **NOT wired into** :class:`NLSSModel` — that is
the integrator's job; this module only provides the semantics + a testable pass.

Role (§04/method.tex, "re-grounding"): when the representation advances
``P_t -> P_{t+1}`` the empirical field that will be rebuilt from it may no
longer be able to *recover every attribute* of each prior evidence record.  Each
prior record must therefore be re-grounded under the NEW contract from its
ORIGINAL language hypothesis (the facade stores ``state.hypotheses[object_id] =
LanguageHypothesis``), reconstructing which attributes the new contract can and
cannot recover.

Missing-data semantics are explicit rather than silent:

  * An attribute the new contract cannot recover is recorded as an
    :class:`UnknownAttribute` (``reason='unrecoverable'``) — it stays
    *unknown*, never fabricated.
  * A record is ``usable_in_inference`` only when **all** required attributes of
    the contract are present **OR** the contract declares valid missing-data
    semantics via ``spec.provenance['missing_data_ok']``.
  * A record that cannot be re-grounded at all (compile/verify/canonicalize
    fails under the new contract) is **excluded from inference** (it is NOT
    deleted from the ledger; re-grounding only reports), or — under the
    ``query`` policy — surfaced for a targeted experiment.

Missing-data policy (module constant :data:`MISSING_DATA_POLICY`)
-----------------------------------------------------------------
``retain`` — keep every re-groundable record in the usable set even when some
required attributes are unknown (trust the original record).
``exclude`` — drop any record whose required attributes are not all recovered
from the inference set (default; conservative and match the paper's "otherwise
the record is EXCLUDED from inference").
``query`` — like ``exclude``, but missing-attribute records are surfaced in
``ReGroundingReport.queried`` for a targeted experiment; they are not usable in
inference until the missing attributes are resolved.

The pass is deterministic, additive, and a pure *consumer* of the adapter
(``compile -> verify -> canonicalize -> descriptor``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .interfaces import TaskAdapter
    from .representation import RepresentationSpecification
    from .state import SolutionSpaceState

# The three missing-data policies (see module docstring).
MISSING_DATA_POLICY = frozenset({"retain", "exclude", "query"})

# Default policy applied by :func:`apply_missing_data_policy`.
DEFAULT_MISSING_DATA_POLICY = "exclude"

# Reason tag for attributes the new contract cannot recover.
UNRECOVERABLE = "unrecoverable"


@dataclass(frozen=True, slots=True)
class UnknownAttribute:
    """One attribute of one evidence record that the new contract cannot recover."""

    record_id: str
    attr: str
    reason: str = UNRECOVERABLE


@dataclass(frozen=True, slots=True)
class ReGroundingReport:
    """Deterministic outcome of one re-grounding pass over the evidence ledger."""

    regrounded: tuple[str, ...]          # records successfully re-grounded under P_{t+1}
    excluded: tuple[str, ...]            # records that cannot be re-grounded (dropped from inference)
    queried: tuple[str, ...]             # records missing required attrs -> targeted experiment
    unknown_attrs: tuple[UnknownAttribute, ...]  # every unrecoverable attribute
    usable_ids: tuple[str, ...]          # records usable in inference (usable_in_inference == True)
    usable_in_inference: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ReGroundingPass:
    """Adapter-consumer pass that re-grounds every prior evidence record under a
    new representation contract ``spec``.

    ``re_ground_ledger`` is async because ``TaskAdapter.compile`` is async; a
    synchronous convenience wrapper :meth:`re_ground_ledger_sync` is provided
    (it fails fast if called inside an already-running event loop).
    """

    def __init__(self, adapter: "TaskAdapter") -> None:
        self.adapter = adapter

    async def re_ground_ledger(
        self,
        state: "SolutionSpaceState",
        spec: "RepresentationSpecification",
    ) -> ReGroundingReport:
        """Re-ground ``state.hypotheses`` under ``spec``.

        For each evidence record (hypothesis) it attempts
        ``adapter.compile -> verify -> canonicalize`` and then reads the
        produced attributes from ``adapter.descriptor(obj)``.  Present /
        unknown attributes are compared against ``spec.typed_variables``.
        """
        required = set(spec.variable_names) if spec is not None else set()
        missing_data_ok = bool(
            (spec.provenance or {}).get("missing_data_ok", False)
        ) if spec is not None else False

        regrounded: list[str] = []
        excluded: list[str] = []
        queried: list[str] = []
        unknown_attrs: list[UnknownAttribute] = []
        usable: list[str] = []
        usable_map: dict[str, bool] = {}

        for record_id, hypothesis in state.hypotheses.items():
            ok, produced = await self._try_reground(hypothesis)
            if not ok:
                # cannot recover this record under the new contract -> exclude.
                excluded.append(record_id)
                usable_map[record_id] = False
                for attr in sorted(required):
                    unknown_attrs.append(UnknownAttribute(record_id, attr, UNRECOVERABLE))
                continue

            regrounded.append(record_id)
            present = set(produced) & required if required else set(produced)
            missing = required - present
            for attr in sorted(missing):
                unknown_attrs.append(UnknownAttribute(record_id, attr, UNRECOVERABLE))

            if not missing or missing_data_ok:
                usable.append(record_id)
                usable_map[record_id] = True
            else:
                queried.append(record_id)  # targeted experiment requested
                usable_map[record_id] = False

        return ReGroundingReport(
            regrounded=tuple(regrounded),
            excluded=tuple(excluded),
            queried=tuple(queried),
            unknown_attrs=tuple(unknown_attrs),
            usable_ids=tuple(usable),
            usable_in_inference=usable_map,
            metadata={
                "missing_data_ok": missing_data_ok,
                "required_attributes": tuple(sorted(required)),
            },
        )

    def re_ground_ledger_sync(
        self,
        state: "SolutionSpaceState",
        spec: "RepresentationSpecification",
    ) -> ReGroundingReport:
        """Synchronous wrapper around :meth:`re_ground_ledger`.

        Raises ``TypeError`` if called inside an already-running event loop
        (the caller should await the async method instead).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.re_ground_ledger(state, spec))
        raise TypeError(
            "re_ground_ledger_sync cannot run inside an active event loop; "
            "await re_ground_ledger instead"
        )

    async def _try_reground(self, hypothesis) -> tuple[bool, set[str]]:
        """Attempt compile->verify->canonicalize under the current adapter, and
        return ``(grounded_ok, produced_attribute_names)``."""
        try:
            raw = await self.adapter.compile(hypothesis, None)
            verdict = self.adapter.verify(raw)
            if not verdict.valid:
                if verdict.repaired_object is None:
                    return False, set()
                raw = verdict.repaired_object
                verdict = self.adapter.verify(raw)
                if not verdict.valid:
                    return False, set()
            obj = self.adapter.canonicalize(raw)
        except Exception:
            return False, set()
        try:
            attrs = set(self.adapter.descriptor(obj).keys())
        except Exception:
            attrs = set()
        return True, attrs


def apply_missing_data_policy(
    report: ReGroundingReport,
    policy: str = DEFAULT_MISSING_DATA_POLICY,
) -> set[str]:
    """Return the set of usable object_ids under a missing-data policy.

    ``excluded`` records (cannot be re-grounded) are always dropped.  Under
    ``retain`` missing-attribute records stay usable; under ``exclude`` and
    ``query`` they are dropped (under ``query`` they are left flagged in
    ``report.queried`` for a targeted experiment).
    """
    if policy not in MISSING_DATA_POLICY:
        raise ValueError(
            f"policy must be one of {sorted(MISSING_DATA_POLICY)}, got {policy!r}"
        )
    regrounded = set(report.regrounded)
    queried = set(report.queried)
    excluded = set(report.excluded)
    if policy == "retain":
        return regrounded
    # exclude / query: missing-attribute records are not usable until resolved.
    return regrounded - queried - excluded


__all__ = [
    "ReGroundingPass",
    "ReGroundingReport",
    "UnknownAttribute",
    "MISSING_DATA_POLICY",
    "DEFAULT_MISSING_DATA_POLICY",
    "UNRECOVERABLE",
    "apply_missing_data_policy",
]
