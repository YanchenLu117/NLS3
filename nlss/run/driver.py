"""V7 run-driver — copyable end-to-end execution (EXPERIMENT_PLAN §6, §9).

Assembles an arm and executes it under a budget in a regime, writing the §12.2
artifact set via RunArchiver and returning unified metrics including the §8.1
Recovery-AURC primary and §8.4 resource accounting.

Two regimes (the plan's tags) are modelled; controlled ENFORCES the budget
(queries + token caps + wall time) so arms are resource-matched, native runs the
official system untouched.  Benchmarks swap in each real oracle; the boolean toy
proves the plumbing end to end.  SciExplorer's four arms are enumerated in
``nlss.adapters.sciexplorer.arms``; an official-repo-aware runner selects among
them and passes the smoke gate before admitting a run (pending the official
repo's install — see DATA_MANIFEST.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..baselines import create_baseline
from ..metrics.recovery_aurc import from_state as _aurc_from_state
from ..runout import RunArchiver


class Regime(str, Enum):
    CONTROLLED = "controlled"
    NATIVE = "native"


@dataclass
class Budget:
    query_cap: int = 40
    batch: int = 4
    rounds: int = 3
    token_input_cap: int | None = None
    token_output_cap: int | None = None
    wall_cap_sec: float | None = None


@dataclass
class RunResult:
    regime: str
    arm: str
    reference_size: int
    seed: int
    artifacts: list[str] = field(default_factory=list)
    rounds_done: int = 0
    queries_used: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    budget_hit: bool = False
    error: str | None = None


def _toy_oracle(adapter, obj, round_index: int) -> float:
    text = str(getattr(obj, "canonical_form", ""))
    return 1.0 if "AND" in text.upper() or "high density" in text else 0.0


def _budget_exceeded(budget: Budget, queries: int, o_in: int, o_out: int, wall: float) -> bool:
    if budget.query_cap is not None and queries >= budget.query_cap:
        return True
    if budget.token_input_cap is not None and o_in >= budget.token_input_cap:
        return True
    if budget.token_output_cap is not None and o_out >= budget.token_output_cap:
        return True
    if budget.wall_cap_sec is not None and wall >= budget.wall_cap_sec:
        return True
    return False


def _token_totals(model) -> tuple[int, int]:
    try:
        u = model.token_usage()
        return (int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0))
    except Exception:
        return 0, 0


def execute_nlss_rounds(
    model,
    adapter,
    *,
    budget: Budget,
    run_dir: str | Path,
    regime: Regime = Regime.CONTROLLED,
    reference: Sequence[str] | None = None,
    seed: int = 0,
    hypothesis_pool: Sequence[str] | None = None,
    arm: str = "full_nlss",
    oracle=None,
) -> RunResult:
    from ..core.model import EvidenceRecord
    from ..core.types import LanguageHypothesis, Observation

    res = RunResult(regime=regime.value, arm=arm, seed=seed,
                    reference_size=len(reference or ()))
    ref = set(reference or ())
    t0 = time.time()
    round_index = 0
    try:
        pool = hypothesis_pool or ["x AND y", "x OR y", "NOT x", "x AND NOT y"]
        for i, t in enumerate(pool[: max(1, budget.batch // 2)]):
            model.update([EvidenceRecord(kind="hypothesis",
                                         hypothesis=LanguageHypothesis(hypothesis_id=f"h{i}", text=t))])
        model.recover()
        oracle = oracle or _toy_oracle  # default toy; real bench supplies its own
        with RunArchiver(run_dir, model_id=f"bool-{seed}", seed=seed,
                         regime=regime.value) as arc:
            for _ in range(max(1, budget.rounds)):
                proposal_ids = list(model.state.objects)
                round_index += 1
                for o in [model.state.objects[x] for x in proposal_ids[: budget.batch]]:
                    model.update([EvidenceRecord(kind="observation",
                                                 observation=Observation(object_id=o.object_id,
                                                                          value=oracle(adapter, o, round_index),
                                                                          round_index=round_index))])
                model.recover()
                o_in, o_out = _token_totals(model)
                wall = time.time() - t0
                arc.write_round(model, round_idx=round_index,
                                token_budget=budget.token_output_cap or 1600)
                res.queries_used += min(budget.batch, len(proposal_ids))
                if _budget_exceeded(budget, res.queries_used, o_in, o_out, wall):
                    res.budget_hit = True
                    break
            res.rounds_done = round_index
            res.metrics = {
                "recovery_aurc": _aurc_from_state(model.state, ref),
                "evidence": len(model.evidence_log),
                "n_objects": len(model.state.objects),
                "n_observations": len(model.state.observations),
            }
            res.resources = {"token_usage": dict(model.token_usage()),
                             "wall_time_sec": round(time.time() - t0, 4),
                             "queries": res.queries_used}
            res.artifacts = arc.list_artifacts()
    except Exception as exc:
        res.error = str(exc)
    return res


def execute_baseline_arm(
    baseline_name: str,
    *,
    budget: Budget,
    run_dir: str | Path,
    regime: Regime = Regime.CONTROLLED,
    seed: int = 0,
    pool: Sequence[Sequence[float]] | None = None,
    pool_ids: Sequence[str] | None = None,
    cfg: Mapping[str, Any] | None = None,
    history: Mapping[str, float] | None = None,
    features: Mapping[str, Sequence[float]] | None = None,
    scalars: Mapping[str, float] | None = None,
) -> RunResult:
    res = RunResult(regime=regime.value, arm=baseline_name, seed=seed, reference_size=0)
    t0 = time.time()
    try:
        prop = create_baseline(baseline_name, cfg)
        if history is not None and hasattr(prop, "set_history"):
            prop.set_history(history)
        if features is not None and hasattr(prop, "set_features"):
            prop.set_features(features)
        if scalars is not None and hasattr(prop, "set_scalars"):
            prop.set_scalars(scalars)
        with RunArchiver(run_dir, model_id=f"{baseline_name}-{seed}", seed=seed,
                         regime=regime.value) as arc:
            for r in range(1, max(1, budget.rounds) + 1):
                chosen, scores = prop.select(pool or [[0.0]], ids=pool_ids, k=budget.batch)
                arc.write_jsonl(f"proposals/candidates_round_{r}.jsonl",
                                [{"id": c, "score": float(s)} for c, s in zip(chosen, scores)])
                arc._write(f"controller/decision_round_{r}.json",
                           {"arm": baseline_name, "batch": budget.batch, "seed": seed})
                res.queries_used += len(chosen)
                res.budget_hit = res.budget_hit or (budget.query_cap is not None and res.queries_used >= budget.query_cap)
                if res.budget_hit:
                    break
            res.metrics = {"n_batches": budget.rounds, **prop.info()}
            res.resources = {"wall_time_sec": round(time.time() - t0, 4),
                             "queries": res.queries_used}
            res.artifacts = arc.list_artifacts()
    except Exception as exc:
        res.error = str(exc)
    return res
