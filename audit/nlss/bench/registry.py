"""registry.py — PAPER_LAB unified experiment registry (paper's single data exit).

Every experiment the paper cites MUST be registered here. Rule R1: a result not
in the registry does not exist. Schema:

  {"exp_id":  "S1A_pilot",
   "benchmark": "BENCH_DISCOVERYBENCH",          # single namespace, see bench_map
   "task_subset": "synth.sample30@seed92711",
   "arms": ["full", "fixedmenu_or", "fixedmenu_llm", "open_noaudit"],
   "baseline_set": ["self_ablation_fixedmenu", "hypogetic_2025"],
   "metrics": ["p_nonmenu", "novelty_at_3", "accepted", "discovery_eval"],
   "stats_protocol": "perm92711_holm_single_family",
   "status": "planned|running|pilot_gated|complete|archived",
   "data_path": "paper_lab/runs/S1A_pilot/",
   "cost_report": "paper_lab/costs/S1A_pilot.jsonl",
   "report": "paper_lab/reports/S1A_pilot.md",
   "prereg": "prereg/openform_freeze.md"}

Append-only updates via update(); each update writes a history line.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

REGISTRY_PATH = Path("paper_lab/registry.json")
HISTORY_PATH = Path("paper_lab/registry_history.jsonl")

BENCH_MAP = {
    "BENCH_DISCOVERYBENCH": "data/benchmarks/discoverybench (official clone)",
    "BENCH_HYPOBENCH": "data/benchmarks/hypobench (official clone)",
    "BENCH_SCIENCEAGENTBENCH": "data/benchmarks/scienceagentbench + HF osunlp/ScienceAgentBench verified split",
    # legacy assets: registered for provenance only, never mixed with new faces
    "LEGACY_RB480": "archived earlier-release manifest (frozen)",
    "LEGACY_DB": "archived earlier-release manifest, GLM run set (frozen)",
    "LEGACY_V8_SUITE": "NLSS_V8_latest_c89 (PRE_THEORY_RETROSPECTIVE)",
}


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path: Path = REGISTRY_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"version": 1, "created": _utc(), "experiments": {}}


def save(reg: dict, path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1))


def register(exp: dict, path: Path = REGISTRY_PATH) -> None:
    reg = load(path)
    exp = dict(exp)
    exp.setdefault("registered", _utc())
    exp.setdefault("status", "planned")
    if exp.get("benchmark") not in BENCH_MAP:
        raise ValueError(f"unknown benchmark namespace: {exp.get('benchmark')}")
    reg["experiments"][exp["exp_id"]] = exp
    save(reg, path)
    with open(HISTORY_PATH.with_suffix(HISTORY_PATH.suffix or ".jsonl"), "a") as g:
        g.write(json.dumps({"ts": _utc(), "action": "register",
                            "exp_id": exp["exp_id"]}, ensure_ascii=False) + "\n")


def update(exp_id: str, patch: dict, path: Path = REGISTRY_PATH) -> None:
    reg = load(path)
    if exp_id not in reg["experiments"]:
        raise KeyError(exp_id)
    reg["experiments"][exp_id].update(patch)
    reg["experiments"][exp_id]["updated"] = _utc()
    save(reg, path)
    hist = path.parent / "registry_history.jsonl"
    with open(hist, "a") as g:
        g.write(json.dumps({"ts": _utc(), "action": "update", "exp_id": exp_id,
                            "patch_keys": sorted(patch.keys())},
                           ensure_ascii=False) + "\n")


def get(exp_id: str, path: Path = REGISTRY_PATH) -> dict:
    return load(path)["experiments"][exp_id]


def summary(path: Path = REGISTRY_PATH) -> str:
    reg = load(path)
    lines = [f"PAPER_LAB registry — {len(reg['experiments'])} experiments"]
    for eid, e in sorted(reg["experiments"].items()):
        lines.append(f"  {eid:24s} {e.get('benchmark','?'):26s} {e.get('status','?'):12s} {e.get('data_path','')}")
    return "\n".join(lines)
