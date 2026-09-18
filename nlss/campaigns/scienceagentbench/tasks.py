"""ScienceAgentBench task loading + stratified sampling.

Model-agnostic. Metadata persisted from the HF `verified` split
(SAB_data/verified_metadata.json). Sampler stratifies by domain and by a
coarse complexity gradient so the subset represents the full benchmark instead
of running all 102 tasks.
"""
import json, random
from pathlib import Path
from collections import defaultdict


def load_metadata(meta_path=None):
    meta_path = meta_path or (Path(__file__).resolve().parents[4]
                              / "external" / "repos" / "scienceagentbench"
                              / "SAB_data" / "verified_metadata.json")
    m = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    n = len(m["instance_id"])
    tasks = []
    for i in range(n):
        t = {}
        for c in ("instance_id", "domain", "subtask_categories", "github_name",
                  "task_inst", "domain_knowledge", "dataset_folder_tree",
                  "dataset_preview", "src_file_or_path", "gold_program_name",
                  "output_fname", "eval_script_name"):
            t[c] = m[c][i]
        t["_complexity"] = _complexity(t)
        tasks.append(t)
    return tasks


def _complexity(t):
    """Coarse difficulty proxy: instruction length + dataset tree size + subtask count."""
    return (len(t.get("task_inst") or "")
            + len(t.get("dataset_folder_tree") or "")
            + len(str(t.get("subtask_categories") or "")))


def domain_counts(tasks):
    c = defaultdict(int)
    for t in tasks:
        c[t["domain"]] += 1
    return dict(c)


def stratify(tasks, per_domain=None, seed=0, want=None):
    """Pick a stratified subset across domains and a low/mid/high complexity gradient.

    per_domain: exact per-domain pick (must be <= min size). If None, auto-tune so
    the total is a top-conference-handling size (default ~16).
    want: optional target total; per_domain derived as ceil(want/#domains).
    """
    rng = random.Random(seed)
    by_dom = defaultdict(list)
    for t in tasks:
        by_dom[t["domain"]].append(t)
    if per_domain is None:
        ndom = len(by_dom)
        per_domain = max(2, round((want if want else 16) / ndom))
    picked = []
    for dom, items in by_dom.items():
        items = sorted(items, key=lambda t: t["_complexity"])
        n = min(per_domain, len(items))
        if n == 0:
            continue
        # span the gradient: take evenly-spaced indices across the sorted list
        idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
        picked.extend([items[i] for i in idx])
    rng.shuffle(picked)
    return picked


def split_into_pool(ev) -> None:  # placeholder for eval-loop splits
    return None
