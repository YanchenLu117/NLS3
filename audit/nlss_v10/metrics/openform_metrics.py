"""openform_metrics.py — S1 OpenFormBench metrics (frozen definitions).

p_nonmenu : fraction of candidates judged NON_MENU under the frozen equivalence
            criterion (PREREG §5): NON_MENU ⟺ for EVERY M_j in M_ref, no
            implementation of M_j preserves the candidate's required semantic
            distinctions/relations/operations within pre-frozen tolerance.
            Surface names NEVER count. M_ref = {vector, distribution, graph,
            hierarchy, discrete-combinatorial} (evaluator-only).
novelty@K : among the K decodes of one (task, arm), the fraction of candidates
            pairwise NON-EQUIVALENT (not menu-equivalent to each other).
accepted  : Hybrid Audit outcome == Pass.
executable: all declared operators ran OK under jail (q_exec smoke+battery).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

M_REF = ["vector", "distribution", "graph", "hierarchy",
         "discrete-combinatorial"]


@dataclass
class CandidateVerdict:
    candidate_id: str
    task_id: str
    arm: str
    decode_seed: int
    is_nonmenu: bool
    audit_outcome: str
    all_ops_exec_ok: bool
    equivalent_to: list = field(default_factory=list)  # candidate_ids


def p_nonmenu(verdicts: list, arm: str = None) -> dict:
    pool = [v for v in verdicts if arm is None or v.arm == arm]
    n = len(pool)
    if not n:
        return {"p_nonmenu": None, "n": 0}
    return {"p_nonmenu": round(sum(v.is_nonmenu for v in pool) / n, 4), "n": n}


def novelty_at_k(verdicts: list, k: int, arm: str = None) -> dict:
    """Mean over tasks of (number of pairwise non-equivalent pairs / C(K,2))."""
    pool = [v for v in verdicts if arm is None or v.arm == arm]
    by_task: dict = {}
    for v in pool:
        by_task.setdefault((v.task_id, v.arm), []).append(v)
    fracs = []
    for (t, a), vs in by_task.items():
        vs = vs[:k]
        if len(vs) < 2:
            continue
        pairs = list(itertools.combinations(range(len(vs)), 2))
        noneq = sum(1 for i, j in pairs
                    if vs[j].candidate_id not in vs[i].equivalent_to)
        fracs.append(noneq / len(pairs))
    if not fracs:
        return {"novelty_at_k": None, "k": k, "n_tasks": 0}
    return {"novelty_at_k": round(sum(fracs) / len(fracs), 4), "k": k,
            "n_tasks": len(fracs)}


def accepted_rate(verdicts: list, arm: str = None) -> dict:
    pool = [v for v in verdicts if arm is None or v.arm == arm]
    if not pool:
        return {"accepted": None, "n": 0}
    return {"accepted": round(sum(v.audit_outcome == "Pass" for v in pool) / len(pool), 4),
            "n": len(pool)}


def executable_rate(verdicts: list, arm: str = None) -> dict:
    pool = [v for v in verdicts if arm is None or v.arm == arm]
    if not pool:
        return {"executable": None, "n": 0}
    return {"executable": round(sum(v.all_ops_exec_ok for v in pool) / len(pool), 4),
            "n": len(pool)}


def arm_summary(verdicts: list, arm: str, k: int = 3) -> dict:
    out = {"arm": arm}
    out.update(p_nonmenu(verdicts, arm))
    out.update(novelty_at_k(verdicts, k, arm))
    out.update(accepted_rate(verdicts, arm))
    out.update(executable_rate(verdicts, arm))
    return out
