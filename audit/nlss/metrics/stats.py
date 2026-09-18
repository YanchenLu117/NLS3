"""stats.py — statistics protocol (frozen 2026-09-17).

Official-tables protocol:
  - paired sign-permutation on task-level deltas, seed 92711, 200k resamples;
  - Holm step-down per STAGE family (each S-experiment = one family);
  - bootstrap 95% CI;
  - pilot gate: hard ≥30 before any formal claim.
"""
from __future__ import annotations

import math
import random

SEED = 92711
N_PERM = 200_000
N_BOOT = 10_000
ALPHA = 0.05


def paired_sign_permutation(deltas: list, seed: int = SEED,
                            n: int = N_PERM) -> dict:
    if not deltas or not all(d == d and math.isfinite(d) for d in deltas):
        raise ValueError("paired_sign_permutation: empty or NaN/inf deltas")
    rng = random.Random(seed)
    m = len(deltas)
    point = sum(deltas) / m
    extreme = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in deltas) / m
        if abs(s) >= abs(point) - 1e-12:
            extreme += 1
    return {"n": m, "point_delta": round(point, 6), "p_perm": round(extreme / n, 6)}


def paired_bootstrap_ci(deltas: list, seed: int = SEED + 1,
                        n: int = N_BOOT) -> dict:
    if not deltas:
        raise ValueError("paired_bootstrap_ci: empty deltas")
    rng = random.Random(seed)
    m = len(deltas)
    boots = sorted(sum(deltas[rng.randrange(m)] for _ in range(m)) / m
                   for _ in range(n))
    return {"ci95": [round(boots[int(0.025 * n)], 6),
                     round(boots[int(0.975 * n) - 1], 6)]}


def holm(pvals: dict) -> dict:
    """Holm step-down over the ACTUAL p-vector (lesson learned: never pin other
    families at 1.0 — that degenerates to Bonferroni×index)."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out, prev = {}, 0.0
    for i, k in enumerate(order):
        adj = min(1.0, max(prev, (m - i) * pvals[k]))
        out[k] = {"p_raw": round(pvals[k], 6), "p_holm": round(adj, 6),
                  "significant": adj <= ALPHA}
        prev = adj
    return out


def stage_contrast(pairs: list, label: str) -> dict:
    """pairs = [(a_value, b_value), ...] paired per (task, decode)."""
    deltas = [a - b for a, b in pairs]
    res = paired_sign_permutation(deltas)
    res.update(paired_bootstrap_ci(deltas))
    w = sum(1 for d in deltas if d > 1e-12)
    l = sum(1 for d in deltas if d < -1e-12)
    res["w"] = w
    res["l"] = l
    res["ties"] = len(deltas) - w - l
    res["label"] = label
    return res


def pilot_gate(n_hard: int, min_hard: int = 30) -> dict:
    """Gate discipline: hard-layer ≥30 before formal release."""
    return {"n_hard": n_hard, "min_hard": min_hard,
            "pass": n_hard >= min_hard}
