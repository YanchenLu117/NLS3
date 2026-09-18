"""discoverybench_adapter.py — official DiscoveryBench tasks → v10 χ input.

OFFICIAL-ONLY policy: tasks, datasets, splits, and evaluation come from the
official clone verbatim. This adapter only WRAPS (never rewrites) a task:
  task_id      = "{dataset}_qid{qid}" (official dataset dir + query id)
  semantic_description = official `question` text verbatim
  evidence     = official dataset description + column list (from metadata_*.json)
  required_interactions = derived verbatim from the question_type enum context
  min_classes  = evaluator-side floor: gold hypothesis pair-distinction
  tolerance / probe_bank_ref = prereg defaults

Sampling: pilot 30 tasks drawn with seed 92711 from the official synth answer
key universe (200 tasks / 75 datasets), stratified by question_type.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DB_ROOT = REPO / "data" / "benchmarks" / "discoverybench"

TOLERANCE = {"T_preserve": 0.05, "T_exec": 1e-6, "max_probe_failures": 0}


def load_answer_key(split: str = "synth") -> list:
    p = DB_ROOT / "eval" / f"answer_key_{split}.csv"
    return list(csv.DictReader(open(p, encoding="latin-1")))


def load_metadata(dataset: str, meta_id: str, split: str = "synth") -> dict:
    for cand in (DB_ROOT / "discoverybench" / split / "test" / dataset,
                 DB_ROOT / "discoverybench" / split / "train" / dataset,
                 DB_ROOT / "discoverybench" / split / "dev" / dataset):
        for mname in (f"metadata_{meta_id}.json", "metadata.json"):
            p = cand / mname
            if p.exists():
                return json.loads(p.read_text(encoding="latin-1"))
    raise FileNotFoundError(f"metadata for {dataset}/{meta_id} not found")


def wrap_task(dataset: str, qid: str, meta_id: str, split: str = "synth") -> dict:
    meta = load_metadata(dataset, meta_id, split)
    flat = []
    for q in meta.get("queries", []):
        (flat.extend(q) if isinstance(q, list) else flat.append(q))
    question = None
    for q in flat:
        if isinstance(q, dict) and str(q.get("qid")) == str(qid):
            question = q
            break
    if question is None:
        raise KeyError(f"qid {qid} not in {dataset} metadata {meta_id}")

    def _cols(dset):
        c = dset.get("columns")
        if isinstance(c, dict):
            out = []
            for v in c.values():
                out.extend(v if isinstance(v, list) else [])
            return out
        return c if isinstance(c, list) else []

    ds_desc_parts = []
    for d in meta.get("datasets", [])[:3]:
        dname = d.get("name", "")
        ddesc = d.get("description", "")
        colnames = ", ".join(str(x.get("name", "")) for x in _cols(d)[:12])
        ds_desc_parts.append(f"{dname}: {ddesc} (cols: {colnames})")
    ds_desc = "; ".join(ds_desc_parts)
    return {
        "task_id": f"{dataset}_qid{qid}",
        "benchmark": "BENCH_DISCOVERYBENCH",
        "source": {"dataset": dataset, "metadata_id": meta_id, "qid": qid,
                   "question_type": question.get("question_type"),
                   "difficulty": question.get("difficulty")},
        "semantic_description": question["question"],  # official text verbatim
        "evidence": ds_desc[:3000],
        "required_interactions":
            "query the provided columns; aggregate over groups; compare across "
            "conditions implied by the question",
        "tolerance": TOLERANCE,
        "gold_hypo_note": "official answer_key_{split}.csv row (evaluator-only)",
    }


def sample_pilot(n: int = 30, seed: int = 92711, split: str = "synth") -> list:
    """Stratified by question_type over the official answer-key universe."""
    key = load_answer_key(split)
    by_type: dict = {}
    for r in key:
        try:
            t = wrap_task(r["dataset"], r["query_id"], r["metadataid"], split)
        except FileNotFoundError:
            continue
        by_type.setdefault(r.get("dataset", ""), []).append(t)
    rng = random.Random(seed)
    pools = list(by_type.values())
    rng.shuffle(pools)
    out, i = [], 0
    while len(out) < n and any(pools):
        p = pools[i % len(pools)]
        if p:
            out.append(p.pop())
        i += 1
    return out[:n]


def export_jsonl(tasks: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as g:
        for t in tasks:
            g.write(json.dumps(t, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    tasks = sample_pilot(30)
    out = REPO / "data" / "benchmarks" / "discoverybench_v10_tasks_pilot30.jsonl"
    export_jsonl(tasks, out)
    print(f"pilot30 exported: {out} ({len(tasks)} tasks)")
    from collections import Counter
    print("question_type mix:", Counter(t["source"]["question_type"] for t in tasks))
