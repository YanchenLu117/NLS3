"""V8 P0-c — automatic representation induction (Detail §3.3).

Frozen five-step flow, fully automatic in scored runs:

1. serialize the frozen dossier, visible evidence, previous contract, and the
   registered failure signal;
2. request ONE schema-valid proposal using the frozen induction prompt;
3. run public syntax/type/executability and actor-visible semantic checks;
4. if rejected, permit ONE repair call containing ONLY the actor-visible
   validator report;
5. if repair fails, retain the previous admitted contract; if no previous
   contract exists, terminate under the P0 failure rule.

Both calls count toward the common budget.  The proposer is injected (runtime
wires an LLM backend through ``nlss.llm``; tests inject fakes) — this module
enforces the CALL DISCIPLINE, not the model.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class InductionOutcome(str, Enum):
    ADMITTED = "ADMITTED"
    ADMITTED_AFTER_REPAIR = "ADMITTED_AFTER_REPAIR"
    RETAINED_PREVIOUS = "RETAINED_PREVIOUS"
    TERMINATED_NO_CONTRACT = "TERMINATED_NO_CONTRACT"


@dataclass(frozen=True)
class InductionPrompt:
    """P0-frozen induction prompt (id + content hash recorded in P0)."""

    prompt_id: str
    content_hash: str


@dataclass(frozen=True)
class ValidatorReport:
    passed: bool
    problems: tuple[str, ...] = ()
    actor_visible: Mapping[str, Any] = field(default_factory=dict)

    def actor_visible_report(self) -> dict[str, Any]:
        """The ONLY payload a repair call may receive (Detail §3.3 step 4)."""
        return copy.deepcopy(dict(self.actor_visible))


class PublicValidator(ABC):
    """Public syntax/type/executability + actor-visible semantic checks."""

    validator_id: str

    @abstractmethod
    def check(self, proposal: Mapping[str, Any]) -> ValidatorReport: ...


class InductionProposer(ABC):
    """Injected proposal source.  ``propose`` and ``repair`` are each called at
    most once per induction run; ``repair`` receives the actor-visible report
    only."""

    @abstractmethod
    def propose(self, serialized_context: Mapping[str, Any]) -> Mapping[str, Any]: ...

    @abstractmethod
    def repair(self, actor_visible_report: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ProposalSchemaError(ValueError):
    """Proposal violates the P0-frozen proposal JSON schema."""


def check_proposal_schema(proposal: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    """Minimal P0 schema check: required keys, per-key types, key closed-world.

    ``schema`` shape: {"required": {key: type_or_tuple}, "optional": {…}}.
    """
    problems: list[str] = []
    required: Mapping[str, Any] = schema.get("required", {})
    for key, types in required.items():
        if key not in proposal:
            problems.append(f"missing required key: {key}")
            continue
        value = proposal[key]
        types_t = types if isinstance(types, tuple) else (types,)
        if isinstance(value, bool) and bool not in types_t:
            problems.append(f"key {key}: bool not allowed here")
        elif not isinstance(value, types_t):
            problems.append(f"key {key}: wrong type {type(value).__name__}")
    allowed = set(required) | set(schema.get("optional", {}))
    extra = sorted(set(proposal) - allowed)
    if extra and not schema.get("allow_extra", False):
        problems.append(f"unknown keys: {', '.join(extra)}")
    return problems


@dataclass
class InductionCallBudget:
    """Both calls count toward the common budget (Detail §3.3)."""

    proposal_calls: int = 0
    repair_calls: int = 0

    @property
    def total_calls(self) -> int:
        return self.proposal_calls + self.repair_calls


@dataclass(frozen=True)
class InductionResult:
    outcome: InductionOutcome
    contract: Mapping[str, Any] | None
    call_budget: InductionCallBudget
    first_report: ValidatorReport
    repair_report: ValidatorReport | None
    serialized_context_hash: str


class RepresentationInductor:
    def __init__(
        self,
        proposer: InductionProposer,
        validator: PublicValidator,
        prompt: InductionPrompt,
        proposal_schema: Mapping[str, Any],
    ) -> None:
        self._proposer = proposer
        self._validator = validator
        self._prompt = prompt
        self._schema = dict(proposal_schema)

    @staticmethod
    def _serialize(
        dossier_ref: str,
        visible_evidence: Mapping[str, Any],
        previous_contract: Mapping[str, Any] | None,
        failure_signal: Mapping[str, Any] | None,
        prompt: InductionPrompt,
    ) -> dict[str, Any]:
        import hashlib
        import json

        context = {
            "dossier_ref": dossier_ref,
            "visible_evidence": copy.deepcopy(dict(visible_evidence)),
            "previous_contract": None if previous_contract is None else copy.deepcopy(dict(previous_contract)),
            "failure_signal": None if failure_signal is None else copy.deepcopy(dict(failure_signal)),
            "prompt_id": prompt.prompt_id,
            "prompt_content_hash": prompt.content_hash,
        }
        blob = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {"context": context, "hash": hashlib.sha256(blob.encode("utf-8")).hexdigest()}

    def run(
        self,
        *,
        dossier_ref: str,
        visible_evidence: Mapping[str, Any],
        previous_contract: Mapping[str, Any] | None = None,
        failure_signal: Mapping[str, Any] | None = None,
    ) -> InductionResult:
        budget = InductionCallBudget()
        serialized = self._serialize(dossier_ref, visible_evidence, previous_contract, failure_signal, self._prompt)

        # step 2 — exactly one proposal call
        budget.proposal_calls += 1
        proposal = self._proposer.propose(serialized["context"])
        if not isinstance(proposal, Mapping):
            proposal = {}
        schema_problems = check_proposal_schema(proposal, self._schema)
        # step 3 — public + actor-visible checks
        first_report = (
            self._validator.check(proposal)
            if not schema_problems
            else ValidatorReport(False, tuple(schema_problems), {"schema_problems": schema_problems})
        )
        if first_report.passed:
            return InductionResult(
                InductionOutcome.ADMITTED,
                copy.deepcopy(dict(proposal)),
                budget,
                first_report,
                None,
                serialized["hash"],
            )

        # step 4 — exactly one repair call, actor-visible report ONLY
        budget.repair_calls += 1
        repaired = self._proposer.repair(first_report.actor_visible_report())
        if not isinstance(repaired, Mapping):
            repaired = {}
        schema_problems2 = check_proposal_schema(repaired, self._schema)
        repair_report = (
            self._validator.check(repaired)
            if not schema_problems2
            else ValidatorReport(False, tuple(schema_problems2), {"schema_problems": schema_problems2})
        )
        if repair_report.passed:
            return InductionResult(
                InductionOutcome.ADMITTED_AFTER_REPAIR,
                copy.deepcopy(dict(repaired)),
                budget,
                first_report,
                repair_report,
                serialized["hash"],
            )

        # step 5 — fallback: retain previous contract, else terminate (P0 rule)
        if previous_contract is not None:
            return InductionResult(
                InductionOutcome.RETAINED_PREVIOUS,
                copy.deepcopy(dict(previous_contract)),
                budget,
                first_report,
                repair_report,
                serialized["hash"],
            )
        return InductionResult(
            InductionOutcome.TERMINATED_NO_CONTRACT,
            None,
            budget,
            first_report,
            repair_report,
            serialized["hash"],
        )
