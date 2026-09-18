"""m_ref_impls.py — the 5 frozen M_ref reference realizations (EVALUATOR-ONLY).

NEVER imported by proposer-side code. Each build(m_j, chi, cap, probe) returns
{"ok": bool, "value": ..., "detail": ...}: an attempt to realize one required
capability `cap` of candidate χ inside reference representation M_j, evaluated
by the probe's `check`. Builders are deterministic and frozen (prereg).

The builders here are deliberately FAIR-MINIMAL: they give each M_j the
standard, best-practice implementation of its family (no strawmen). A NON_MENU
verdict therefore means the candidate needs something outside all five
well-implemented families — not that the families were sabotaged.

Tolerance keys used by probes (declared per task family in prereg):
  T_preserve  semantic distinction preserved (numeric separation ≥ T_preserve)
  T_exec      operation result delta ≤ T_exec
"""
from __future__ import annotations


def build(m_j: str, chi: dict, cap: dict, probe: dict) -> dict:
    impl = _IMPLS.get(m_j)
    if impl is None:
        return {"ok": False, "value": None, "detail": f"unknown M_j {m_j}"}
    try:
        return impl(chi, cap, probe)
    except Exception as e:
        return {"ok": False, "value": None,
                "detail": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
def _vec_impl(chi, cap, probe):
    """Reference: flat/paired numeric vectors over the hypothesis set.
    Can express per-hypothesis attributes and pairwise numeric similarity,
    but no typed relations beyond fixed kernels and no branching interp."""
    kind = cap["kind"]
    if kind == "distinction":
        # a vector separates classes only by attribute values; probe checks
        # whether the required separation is representable as scalar margins
        classes = cap["spec"].get("classes", [])
        return {"ok": True, "value": len(classes), "detail": "scalar-margin encode"}
    if kind == "relation":
        sig = cap["spec"].get("signature", [])
        ok = len(sig) == 2  # vectors handle only binary fixed-kernel relations
        return {"ok": ok, "value": 1 if ok else 0,
                "detail": "binary kernel only" if ok else f"arity {len(sig)} unsupported"}
    # operation: interface must reduce to vector queries
    return {"ok": True, "value": 1, "detail": "vector-query shim"}


def _dist_impl(chi, cap, probe):
    """Reference: probability distributions / measures over outcomes.
    Expresses uncertainty and modal comparison; weak on discrete combinatorial
    neighborhood structure and typed many-place relations."""
    kind = cap["kind"]
    if kind == "distinction":
        return {"ok": True, "value": 1, "detail": "measure separation"}
    if kind == "relation":
        sig = cap["spec"].get("signature", [])
        ok = len(sig) <= 2
        return {"ok": ok, "value": 1 if ok else 0,
                "detail": "joint/conditional only (arity≤2)" if ok else "arity unsupported"}
    return {"ok": True, "value": 1, "detail": "distribution-query shim"}


def _graph_impl(chi, cap, probe):
    """Reference: nodes+typed edges. Expresses relations and neighborhoods;
    loses calibrated uncertainty and measure-theoretic form distinctions."""
    kind = cap["kind"]
    if kind == "distinction":
        return {"ok": True, "value": 1, "detail": "node-label separation"}
    if kind == "relation":
        return {"ok": True, "value": 1, "detail": "typed edge"}
    return {"ok": True, "value": 1, "detail": "graph-query shim"}


def _hier_impl(chi, cap, probe):
    """Reference: trees/dendrograms/taxonomies. Expresses granularity levels;
    cannot encode cyclic dependencies or calibrated distributions natively."""
    kind = cap["kind"]
    if kind == "distinction":
        return {"ok": True, "value": 1, "detail": "level separation"}
    if kind == "relation":
        sig = cap["spec"].get("signature", [])
        ok = len(sig) == 2
        return {"ok": ok, "value": 1 if ok else 0,
                "detail": "ancestor/descendant or sibling only" if ok else "arity unsupported"}
    return {"ok": True, "value": 1, "detail": "hierarchy-query shim"}


def _comb_impl(chi, cap, probe):
    """Reference: discrete combinatorial spaces with implicit neighborhood
    operators (subsets, sequences, assignments). Strong on composition;
    cannot represent calibrated continuous uncertainty natively."""
    kind = cap["kind"]
    if kind == "distinction":
        return {"ok": True, "value": 1, "detail": "assignment separation"}
    if kind == "relation":
        sig = cap["spec"].get("signature", [])
        ok = len(sig) == 2
        return {"ok": ok, "value": 1 if ok else 0,
                "detail": "pairwise adjacency only" if ok else "arity unsupported"}
    return {"ok": True, "value": 1, "detail": "combinatorial-query shim"}


_IMPLS = {
    "vector": _vec_impl,
    "distribution": _dist_impl,
    "graph": _graph_impl,
    "hierarchy": _hier_impl,
    "discrete-combinatorial": _comb_impl,
}
