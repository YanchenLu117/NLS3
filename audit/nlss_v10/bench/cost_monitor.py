"""cost_monitor.py — per-call token cost tracking + budget circuit breaker.

Every LLM call (proposer or judge) goes through track_call(). Writes append-only
JSONL per experiment: paper_lab/costs/<exp_id>.jsonl

  {"ts", "exp_id", "arm", "role": proposer|judge, "model",
   "in_tokens", "out_tokens", "latency_s", "task_id", "decode_seed"}

Budget rules (prereg-frozen per experiment):
  per_task_budget   max output tokens per (task, arm, decode) — breaker trips
                    BudgetExceeded, cell marked, no silent overrun.
  exp_budget_total  soft cap for the whole experiment — crossing it pauses
                    dispatch (raise BudgetPause) until operator raises cap via
                    ledger entry (append-only deviation).
Summary per arm for reports: sum/mean/p95.
"""
from __future__ import annotations

import datetime
import json
import threading
from collections import defaultdict
from pathlib import Path

COST_DIR = Path("paper_lab/costs")
_lock = threading.Lock()


class BudgetExceeded(Exception):
    pass


class BudgetPause(Exception):
    pass


def track_call(exp_id: str, role: str, model: str, in_tokens: int,
               out_tokens: int, latency_s: float, task_id: str = "",
               arm: str = "", decode_seed: int = -1) -> dict:
    rec = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "exp_id": exp_id, "role": role, "model": model,
           "in_tokens": int(in_tokens), "out_tokens": int(out_tokens),
           "latency_s": round(float(latency_s), 2),
           "task_id": task_id, "arm": arm, "decode_seed": decode_seed}
    p = COST_DIR / f"{exp_id}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(p, "a") as g:
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def task_cost(exp_id: str, task_id: str) -> dict:
    """Cumulative cost for one task cell — checked against per-task budget."""
    p = COST_DIR / f"{exp_id}.jsonl"
    if not p.exists():
        return {"in": 0, "out": 0}
    inn = out = 0
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("task_id") == task_id:
            inn += r.get("in_tokens", 0)
            out += r.get("out_tokens", 0)
    return {"in": inn, "out": out}


def check_budget(exp_id: str, task_id: str, per_task_budget: int,
                 exp_budget_total: int | None = None) -> None:
    c = task_cost(exp_id, task_id)
    if c["out"] > per_task_budget:
        raise BudgetExceeded(
            f"{exp_id}/{task_id}: output {c['out']} > per_task_budget {per_task_budget}")
    if exp_budget_total is not None:
        tot = sum(task_cost(exp_id, t)["out"] for t in _all_tasks(exp_id))
        if tot > exp_budget_total:
            raise BudgetPause(f"{exp_id}: total output {tot} > exp_budget_total {exp_budget_total}")


def _all_tasks(exp_id: str) -> set:
    p = COST_DIR / f"{exp_id}.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l).get("task_id") for l in open(p) if l.strip()}


def summary(exp_id: str) -> dict:
    p = COST_DIR / f"{exp_id}.jsonl"
    if not p.exists():
        return {}
    by_arm = defaultdict(lambda: {"in": 0, "out": 0, "calls": 0, "latency": []})
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        a = by_arm[r.get("arm") or "?"]
        a["in"] += r.get("in_tokens", 0)
        a["out"] += r.get("out_tokens", 0)
        a["calls"] += 1
        a["latency"].append(r.get("latency_s", 0.0))
    out = {}
    for a, v in by_arm.items():
        lat = sorted(v.pop("latency"))
        v["latency_mean_s"] = round(sum(lat) / len(lat), 1) if lat else 0
        v["latency_p95_s"] = lat[int(0.95 * len(lat))] if lat else 0
        out[a] = v
    return out
