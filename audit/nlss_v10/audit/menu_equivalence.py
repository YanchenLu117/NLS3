"""menu_equivalence.py — the NON_MENU criterion evaluator (PREREG §5, frozen).

For each M_j ∈ M_ref = {vector, distribution, graph, hierarchy,
discrete-combinatorial} there is ONE frozen reference implementation M_j_ref.
A candidate Form is MENU-EQUIVALENT to M_j iff some use of M_j_ref preserves
the candidate's required semantic distinctions, relations, and operation
capabilities within the pre-frozen tolerance T (declared per task family in
prereg). Judgment is by PROBE BEHAVIOR ONLY — the candidate's self-description
and surface names are never evidence.

Procedure (all traces logged):
  1. collect the candidate's required capabilities R = required_distinctions
     ∪ relations ∪ exploration_interfaces (compiled by the auditor);
  2. for each M_j: attempt to build an M_j_ref realization of χ satisfying R
     (deterministic builder + fixed probe battery, tolerance T);
  3. NON_MENU ⟺ ∀M_j the attempt fails at least one probe beyond T.
Evaluator-only: candidate-side code never runs here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..schema.op_executor import run_operator

M_REF = ["vector", "distribution", "graph", "hierarchy",
         "discrete-combinatorial"]


@dataclass
class EquivAttempt:
    m_j: str
    probes_run: int
    probes_failed: int
    within_tolerance: bool
    trace: str


def compile_required_capabilities(chi: dict) -> list:
    """R = the capability list the candidate demands (evaluator-side compile)."""
    caps = []
    for i, rd in enumerate(chi.get("content", {}).get("required_distinctions") or []):
        caps.append({"cap_id": f"RD-{i:02d}", "kind": "distinction",
                     "spec": rd})
    for i, r in enumerate(chi.get("content", {}).get("relations") or []):
        caps.append({"cap_id": f"REL-{i:02d}", "kind": "relation", "spec": r})
    for i, ei in enumerate(chi.get("form", {}).get("exploration_interfaces") or []):
        caps.append({"cap_id": f"OP-{i:02d}", "kind": "operation", "spec": ei})
    return caps


def try_realization(m_j: str, chi: dict, probe_battery: list,
                    tolerance: dict) -> EquivAttempt:
    """Attempt to realize χ's required capabilities inside reference M_j_ref.

    probe_battery: [{"probe_id", "capability", "check", "tol_key"}...]
    Deterministic builders per M_j live in the evaluator-only module
    m_ref_impls (frozen); this function drives them through probes.
    """
    from .m_ref_impls import build  # frozen reference implementations
    caps = compile_required_capabilities(chi)
    run, failed = 0, 0
    traces = []
    for pb in probe_battery:
        cap = next((c for c in caps if c["cap_id"] == pb["capability"]), None)
        if cap is None:
            continue
        run += 1
        res = build(m_j, chi, cap, pb)
        ok = bool(res.get("ok")) and _within(res, tolerance, pb)
        if not ok:
            failed += 1
        traces.append({"probe_id": pb["probe_id"], "ok": ok,
                       "detail": json.dumps(res, ensure_ascii=False)[:200]})
    t_ok = failed <= tolerance.get("max_probe_failures", 0)
    return EquivAttempt(m_j=m_j, probes_run=run, probes_failed=failed,
                        within_tolerance=t_ok,
                        trace=json.dumps(traces, ensure_ascii=False)[:2000])


def _within(res: dict, tolerance: dict, pb: dict) -> bool:
    """Pre-frozen tolerance check: numeric deltas ≤ T[key], else strict."""
    if not res.get("ok"):
        return False
    got = res.get("value")
    want = pb.get("expect")
    if want is None:
        return True
    key = pb.get("tol_key")
    if key and key in tolerance and isinstance(got, (int, float)) \
       and isinstance(want, (int, float)):
        return abs(got - want) <= tolerance[key]
    return got == want


def judge_nonmenu(chi: dict, probe_battery: list, tolerance: dict,
                  m_ref: list = None) -> dict:
    """Top entry: returns {'is_nonmenu': bool, 'menu_equivalent_to': [...],
    'attempts': [...]}. NON_MENU ⟺ every M_j attempt fails tolerance."""
    attempts = []
    equiv = []
    for m_j in (m_ref or M_REF):
        att = try_realization(m_j, chi, probe_battery, tolerance)
        attempts.append(att)
        if att.within_tolerance:
            equiv.append(m_j)
    return {"is_nonmenu": not equiv, "menu_equivalent_to": equiv,
            "attempts": [vars(a) for a in attempts]}


# ------------------------------------------------------------------ #
# M_ref+ strong closure (PREREG_V10_H0_freeze §2, frozen 2026-09-17) #
# ------------------------------------------------------------------ #

FAMILY_ENVELOPE = {
    # capability-requirement tags each family CANNOT express (frozen)
    "vector": {"typed-relations", "cyclic-dependencies",
               "combinatorial-neighborhood", "measure-theoretic-form"},
    "distribution": {"typed-relations", "combinatorial-neighborhood"},
    "graph": {"calibrated-uncertainty", "measure-theoretic-form"},
    "hierarchy": {"cyclic-dependencies", "typed-relations",
                  "calibrated-uncertainty", "measure-theoretic-form"},
    "discrete-combinatorial": {"calibrated-uncertainty",
                               "measure-theoretic-form"},
}
FAMILY_MAX_ARITY = {
    "vector": 2, "distribution": 2, "graph": 99,
    "hierarchy": 2, "discrete-combinatorial": 2,
}


def capability_needs(cap: dict) -> dict:
    """Declared structural demands of one required capability.

    Read ONLY from spec-declared fields (no NLP, no LLM): signature arity,
    explicit `requires` tags, `calibrated` flag, relation type composition
    (n-ary functional composition uses its full signature).
    """
    spec = cap.get("spec") or {}
    sig = spec.get("signature") or []
    if not isinstance(sig, list):
        sig = []
    req = set(spec.get("requires") or [])
    if spec.get("calibrated") is True:
        req.add("calibrated-uncertainty")
    kind = cap.get("kind")
    arity = len(sig)
    if kind == "relation" and spec.get("type") == "composition" and arity < 2:
        arity = 2
    return {"kind": kind, "arity": arity, "requires": sorted(req)}


def family_supports(m_j: str, needs: dict) -> bool:
    """Frozen envelope check: does family m_j express this capability?"""
    if needs.get("kind") == "operation":
        return True  # exploration interfaces are shimmable in every family
    if needs["arity"] > FAMILY_MAX_ARITY[m_j]:
        return False
    return not (set(needs["requires"]) & FAMILY_ENVELOPE[m_j])


def judge_nonmenu_plus(chi: dict) -> dict:
    """Strong-closure NON_MENU judgment over declared capability structure.

    Returns families that cover the full set, hybrid pairs that cover it
    (capability-assignment split + composition shim), and the strong verdict.
    Evaluator-only, deterministic, no candidate code executed.
    """
    caps = compile_required_capabilities(chi)
    needs = [capability_needs(c) for c in caps]
    if not needs:
        return {"is_nonmenu_strong": False, "menu_equivalent_to": list(M_REF),
                "hybrid_equivalent_to": [], "n_caps": 0,
                "support_matrix": {}, "note": "no declared capabilities"}
    fam_cover = {m: all(family_supports(m, n) for n in needs) for m in M_REF}
    fam_ok = [m for m, ok in fam_cover.items() if ok]
    pairs = [(a, b) for i, a in enumerate(M_REF) for b in M_REF[i + 1:]]
    hybrid_ok = []
    for a, b in pairs:
        if all(family_supports(a, n) or family_supports(b, n) for n in needs):
            hybrid_ok.append(f"{a}x{b}")
    # per-capability support matrix (which families realize each cap)
    matrix = []
    for c, n in zip(caps, needs):
        matrix.append({"cap_id": c["cap_id"], "kind": n["kind"],
                       "arity": n["arity"], "requires": n["requires"],
                       "supported_by": [m for m in M_REF if family_supports(m, n)]})
    return {"is_nonmenu_strong": not fam_ok and not hybrid_ok,
            "menu_equivalent_to": fam_ok,
            "hybrid_equivalent_to": hybrid_ok,
            "n_caps": len(caps),
            "support_matrix": matrix}
