"""V8 — campaign harness: the runtime realization wiring (Detail §2.3/§3/§6).

Composes the V8 modules into a runnable campaign loop:

    prereg assert_scoreable → per-checkpoint rounds (RoundInputs → method →
    RoundOutputs) → oracle discipline (legal unqueried only; illegal/duplicate
    proposals consume a batch slot but no label) → R_j via §6 conventions →
    artifact writing (§15 tree) → resource meter → final Recovery-AUC.

The acting method is injected (LLM-agnostic); the toy oracle in this module
proves the loop end to end without any model.  This is the layer the real
benchmarks plug into.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any, Protocol

from .admission import RecoveryPrediction, RoundInputs, RoundOutputs, RemainingBudgets, StoppingRule, TaskDossier
from .artifacts import EvidenceLedger, RunArtifactWriter, RunManifestBuilder
from .fairness import ResourceMeter
from .metrics import MetricsProtocolError, checkpoint_recovery, recovery_aurc, utility_auc
from ..prereg import BlockedCellError, PreregRegistry, load_registry


class ToyOracle:
    """Minimal §2.1-faithful oracle: answers only legal, unqueried candidates.

    Duplicate/illegal/unavailable proposals consume a batch slot but receive NO
    oracle label (Detail §2.3); vacancies are recorded."""

    def __init__(
        self,
        legal_ids: tuple[str, ...],
        observations: Mapping[str, float],
        solution_ids: set[str],
        initial_ids: set[str],
    ) -> None:
        self.legal = set(legal_ids)
        self.observations = dict(observations)
        self.solution = set(solution_ids)
        self.queried: set[str] = set(initial_ids)
        self.rejected: list[str] = []

    def observe(self, candidate_ids: Sequence[str]) -> tuple[dict[str, float], list[str], list[str]]:
        answered: dict[str, float] = {}
        illegal: list[str] = []
        duplicates: list[str] = []
        for cid in candidate_ids:
            if cid not in self.legal:
                illegal.append(cid)
                continue
            if cid in self.queried:
                duplicates.append(cid)
                continue
            if cid not in self.observations:
                illegal.append(cid)  # unmeasured candidate — no label exists
                continue
            self.queried.add(cid)
            answered[cid] = self.observations[cid]
        return answered, illegal, duplicates


class ActingMethod(Protocol):
    """The acting process (Detail §2.1): proposes candidates from the round inputs."""

    def round_step(self, inputs: RoundInputs, history: Mapping[str, float]) -> RoundOutputs: ...


@dataclass
class CheckpointRecord:
    round_index: int
    budget: int
    recovery: float
    depleted: bool
    best_utility: float
    n_answered: int
    n_illegal: int
    n_duplicates: int


@dataclass
class CampaignResult:
    checkpoints: list[CheckpointRecord]
    recovery_auc: float
    utility_auc: float
    complete: bool
    cap_violations: list[str]


def run_campaign(
    *,
    prereg: PreregRegistry | Mapping[str, Any],
    cell_id: str,
    method: ActingMethod,
    oracle: ToyOracle,
    budgets: Sequence[int],
    run_dir,
    initial_ids: set[str],
    caps: Mapping[str, float] | None = None,
    certify: Callable[[], Any] | None = None,
    seed: int = 0,
) -> CampaignResult:
    """Run one controlled campaign to the preregistered checkpoints.

    ``prereg`` may be a built registry or a published document path/dict; the
    cell must be scoreable (Detail §2.2) before round one."""
    if isinstance(prereg, PreregRegistry):
        registry = prereg
    elif isinstance(prereg, Mapping):
        registry = PreregRegistry.from_published(prereg)
    else:
        registry = PreregRegistry.from_published(load_registry(str(prereg)))
    registry.assert_scoreable(cell_id)

    writer = RunArtifactWriter(run_dir)
    ledger = EvidenceLedger(run_dir / "evidence" / "ledger.jsonl")
    meter = ResourceMeter(caps) if caps else None

    dossier = TaskDossier(
        task_id=cell_id,
        public_description_ref="dossier://campaign",
        validator_interface_ref="validators://campaign",
        initial_evidence_ids=tuple(sorted(initial_ids)),
        legality_rule_id="legal_unqueried_only",
    )
    stopping = StoppingRule("budget_exhausted", "stopping://campaign")

    records: list[CheckpointRecord] = []
    cap_violations: list[str] = []
    history: dict[str, float] = {cid: oracle.observations[cid] for cid in sorted(initial_ids) if cid in oracle.observations}
    complete = True

    for index, budget in enumerate(budgets):
        remaining_oracle = budget - len(oracle.queried)
        if meter is not None:
            remaining_resource = {key: float(meter.caps[key]) - meter.used[key] for key in meter.caps}
        else:
            remaining_resource = {}
        inputs = RoundInputs(
            task_dossier=dossier,
            current_evidence=tuple(ledger.records()),
            legal_candidate_ids=tuple(sorted(oracle.legal - oracle.queried)),
            remaining_budgets=RemainingBudgets(oracle=max(0, remaining_oracle), resource=remaining_resource),
            stopping_rule=stopping,
        )
        outputs = method.round_step(inputs, history)

        # oracle discipline: only legal unqueried candidates get labels
        proposals = tuple(outputs.proposed_candidate_ids)[: max(0, remaining_oracle)]
        answered, illegal, duplicates = oracle.observe(proposals)
        ledger.append([{"round": index, "cid": cid, "y": y} for cid, y in answered.items()])
        for cid, y in answered.items():
            history[cid] = y
        if meter is not None:
            meter.consume("llm_calls", 1)
            meter.consume("oracle_budget", len(answered))
            if meter.exceeded["oracle_budget"]:
                cap_violations.append(f"oracle_budget exceeded at checkpoint {index}")

        # §6 common-universe recovery at this checkpoint.  A missing/empty
        # prediction receives the P0-registered failed-checkpoint score (§2.3).
        universe = (oracle.legal - initial_ids) - oracle.queried
        prediction = outputs.recovery_prediction
        if prediction.missing or (prediction.scores is None and not prediction.submitted_set):
            recovery, depleted = 0.0, False  # registered failed-checkpoint score
        else:
            recovery, depleted = checkpoint_recovery(
                universe=universe,
                solution_ids=oracle.solution,
                scores=dict(prediction.scores) if prediction.scores else None,
                submitted_set=tuple(prediction.submitted_set) if prediction.submitted_set else None,
            )
        best_utility = max((history.get(cid, 0.0) for cid in oracle.solution), default=0.0)
        records.append(
            CheckpointRecord(
                round_index=index,
                budget=budget,
                recovery=recovery,
                depleted=depleted,
                best_utility=best_utility,
                n_answered=len(answered),
                n_illegal=len(illegal),
                n_duplicates=len(duplicates),
            )
        )
        writer.write(
            "metrics_round",
            {"round": index, "primary_endpoint": "recovery", "primary_value": recovery},
            round_t=index,
        )
        writer.write(
            "resources_usage",
            {"round": index, "llm_calls": 1, "tokens_in": 0, "tokens_out": 0},
            round_t=index,
        )
        if meter is not None and meter.exceeded:
            complete = False  # §13.1: incomplete-under-cap remains the primary result

    recs = [r.recovery for r in records]
    utils = [r.best_utility for r in records]
    try:
        auc = recovery_aurc(budgets, recs)
        util = utility_auc(budgets, utils)
    except MetricsProtocolError:
        auc, util, complete = 0.0, 0.0, False

    # §15 finalize: run manifest + content-addressed artifact manifest
    content = registry._document.get("content", {}) if isinstance(registry, PreregRegistry) else {}
    # §15: the achieved certification level and narrow flag come from a real
    # certify_level2 report (AC precondition a) — never hand-filled.  When no
    # evaluator certification is supplied, the level is operationally 0.
    certification_report = certify() if certify is not None else None
    achieved_level = int(certification_report.achieved_level) if certification_report is not None else 0
    narrow = bool(certification_report.narrow) if certification_report is not None else False
    if certification_report is not None:
        writer.write(
            "representation_certificate",
            {
                "version": 1,
                "achieved_level": achieved_level,
                "narrow": narrow,
                "conditions": dict(certification_report.conditions),
                "problems": list(certification_report.problems),
                "stats": dict(certification_report.stats),
            },
            version=1,
        )

    manifest = (
        RunManifestBuilder()
        .set("p0_registry_hash", registry.published_hash or "unpublished")
        .set("commits", content.get("repository", {}).get("commits", {}))
        .set("environment", "campaign-harness")
        .set("model_config_ids", [content.get("model", {}).get("model_id", "unknown")])
        .set("prompts", content.get("prompts", {}).get("prompt_hashes", {}))
        .set("seed", seed)
        .set("budgets", {"checkpoints": list(budgets)})
        .set("thresholds", content.get("solutions", {}).get("thresholds", {"_": "unset"}))
        .set("baseline_roster", content["cells"].get(cell_id, {}).get("baseline_ids", []))
        .set("result_visibility_time", "POST_CAMPAIGN")
        .set("representation_checkpoint_map", {"0": 1} if certification_report is not None else {})
        .set("achieved_certification_level", achieved_level)
        .set("terminal_status", "COMPLETE" if complete else "INCOMPLETE_UNDER_CAP")
        .set("na_reasons", {})
        .build()
    )
    manifest = {**manifest, "certification_narrow": narrow}
    writer.write_manifest(manifest)

    return CampaignResult(
        checkpoints=records,
        recovery_auc=auc,
        utility_auc=util,
        complete=complete and not cap_violations,
        cap_violations=cap_violations,
    )
