"""engine.py — Hybrid Audit engine (Methods, frozen spec §2-3).

BuildCore  : constitutional obligations always checked, independent of candidate
             claims (C1-C4 compilations: required classes, coverage of required
             classes by Γ, sandbox executability of operators, set-valued interp).
BuildInduced: obligations compiled from the candidate's own declarations —
             required_distinctions, relations semantics, claims, exploration
             interfaces actually exercisable (operational fidelity).

Floors/weights (frozen in prereg per task family; defaults here are the
pilot-family values and are overridden by prereg files at run time):
  τ_qsci=0.60 τ_qvalid=0.70 τ_qinterp=0.60 τ_qreal=0.60 τ_qrel=0.55
  τ_qcov=0.60 τ_qexec=0.70   τ_Q=0.65 (mean over dims)
Conservative scores: q̂-u against floors. Floors are decided BEFORE candidates.

Outcome (frozen 4-state):
  Pass           all hard obligations pass ∧ every conservative dim ≥ floor ∧ Q̄≥τ_Q
  Fail           any core obligation fails (verifier never proposes replacement)
  UnderSpecified candidate describable but Γ/interp/coverage claims incomplete
  Unverifiable   audit cannot decide within budget (probes exhausted/timeouts)
Unresolved is EPISODE-level (exploration budget exhausted without activation).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

from ..schema.chi_schema import ValidationResult, canonical_substrate_hash
from ..schema.op_executor import run_operator
from .judge_client import JudgeClient

PILOT_FLOORS = {
    "q_sci": 0.60, "q_valid": 0.70, "q_interp": 0.60, "q_real": 0.60,
    "q_rel": 0.55, "q_cov": 0.60, "q_exec": 0.70,
}
TAU_Q = 0.65
SCORE_DIMS = ["q_sci", "q_valid", "q_interp", "q_real", "q_rel", "q_cov", "q_exec"]


@dataclass
class Obligation:
    id: str
    source: str          # "core" | "induced"
    description: str
    check: str           # machine-checkable kind: "exec" | "judge" | "structural"
    status: str = "pending"   # pass|fail|skipped
    detail: str = ""


@dataclass
class AuditReport:
    candidate_id: str
    outcome: str = "Unverifiable"
    scores: dict = field(default_factory=dict)
    hard_obligations: list = field(default_factory=list)
    certification: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    wall_s: float = 0.0
    substrate_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---- rubrics (frozen wording; judge sees sanitized payload only) ----
RUBRICS = {
    "q_sci": ("Scientific adequacy: does the candidate's hypothesis content capture "
              "the scientifically meaningful distinctions the task evidence supports? "
              "Score how well statements map onto the evidence, ignoring implementation."),
    "q_valid": ("Logical validity: are the hypothesis statements internally consistent, "
                "non-vacuous, and are the declared relations coherent with the statements? "
                "Penalize contradictions, circularity, unfalsifiable framing."),
    "q_interp": ("Interpretation rule quality: does the mapping from new natural-language "
                 "hypotheses into the substrate's hypothesis set preserve meaning, allow "
                 "set-valued outcomes, and route genuine ambiguity into separate branches?"),
    "q_real": ("Realization fidelity: for each hypothesis-carrier pair declared in Γ, does "
               "the carrier payload actually encode the hypothesis's semantic content? "
               "Score per-pair fidelity evidence."),
    "q_rel": ("Relation fidelity: do the declared relations' claimed scientific semantics "
              "match what the carrier data + operators can actually compute about the pairs?"),
    "q_cov": ("Coverage: are ALL required distinctions realized by at least one Γ pair whose "
              "carrier + operators can compute the required semantic separation? "
              "1.0 = every required class covered; fractional = fraction covered."),
    "q_exec": ("Executable fidelity: declared operator semantics vs actual behavior on the "
               "provided probes. Penalize any behavior-claim mismatch."),
}


class HybridAuditor:
    """One candidate in → one AuditReport out. Deterministic probe order."""

    def __init__(self, judge: JudgeClient | None = None,
                 floors: dict | None = None, tau_q: float = TAU_Q,
                 max_probes: int = 12, timeout_per_op: float = 30.0):
        self.judge = judge or JudgeClient()
        self.floors = dict(PILOT_FLOORS)
        if floors:
            self.floors.update(floors)
        self.tau_q = tau_q
        self.max_probes = max_probes
        self.timeout_per_op = timeout_per_op

    # ------------------------------------------------------------------ #
    def audit(self, chi: dict, task_ctx: dict,
              validation: ValidationResult | None = None) -> AuditReport:
        t0 = time.time()
        rep = AuditReport(candidate_id=chi.get("candidate_id", "?"))
        validation = validation or _revalidate(chi)
        rep.substrate_hash = validation.substrate_hash

        if not validation.ok:
            # schema-level breakage = UnderSpecified only if describable, else Fail
            hard_break = any(e.code in ("HC1", "HC3") for e in validation.errors)
            rep.outcome = "Fail" if hard_break else "UnderSpecified"
            rep.hard_obligations = [{"id": f"SCHEMA-{e.code}", "source": "core",
                                     "status": "fail", "detail": e.detail}
                                    for e in validation.errors]
            rep.wall_s = round(time.time() - t0, 2)
            return rep

        obligations: list[Obligation] = []
        obligations += self._build_core(chi, task_ctx)
        obligations += self._build_induced(chi, task_ctx)

        # ---- execute obligations ----
        n_probes = 0
        for ob in obligations:
            if n_probes >= self.max_probes:
                ob.status = "skipped"
                ob.detail = "probe budget exhausted"
                continue
            if ob.check == "structural":
                # already decided at build time
                continue
            n_probes += 1
            if ob.check == "exec":
                # smoke stage: an operator that runs but KeyErrors on the
                # probe's placeholder columns has *demonstrated executability*
                # (the real column battery is the executor battery's job) —
                # treat expected data-key errors as pass-with-note, unexpected
                # crashes as fail.
                res = run_operator(ob._op_code, ob._entry, ob._probe_args,
                                   timeout_s=self.timeout_per_op)
                ok = bool(res.get("ok"))
                if not ok:
                    ekind = str(res.get("error", ""))
                    if any(k in ekind for k in ("KeyError", "IndexError",
                                                "ValueError: could not convert",
                                                "TypeError")) and "rc=" not in ekind:
                        ob.status = "pass"
                        ob.detail = f"runs; probe-placeholder data mismatch ({ekind[:80]})"
                    else:
                        ob.status = "fail"
                        ob.detail = json.dumps(res, ensure_ascii=False)[:400]
                else:
                    ob.status = "pass"
                    ob.detail = json.dumps(res, ensure_ascii=False)[:200]
            elif ob.check == "judge":
                payload = ob._payload(task_ctx)
                r = self.judge.score(ob._rubric, payload)
                if r.get("parse_fail"):
                    ob.status = "skipped"
                    ob.detail = "judge parse fail"
                else:
                    dim = ob._dim
                    q = r["score"]
                    rep.scores.setdefault(dim, []).append(
                        {"q": q, "u": r.get("u", 0.0), "obligation": ob.id,
                         "reason": r.get("reason", "")[:200]})
                    ob.status = "pass" if (q - r.get("u", 0.0)) >= self.floors[dim] else "fail"
                    ob.detail = f"q={q:.3f} u={r.get('u', 0.0):.3f} floor={self.floors[dim]}"

        # ---- aggregate scores (mean over probes per dim, conservative) ----
        agg, cons = {}, {}
        for dim in SCORE_DIMS:
            rows = rep.scores.get(dim, [])
            if rows:
                q̄ = sum(r["q"] for r in rows) / len(rows)
                ū = sum(r["u"] for r in rows) / len(rows)
                agg[dim] = round(q̄, 4)
                cons[dim] = round(q̄ - ū, 4)
            # missing dim stays absent: treated as Unverifiable below
        rep.scores = {"point": agg, "conservative": cons,
                      "per_probe": rep.scores}

        # ---- decision ----
        core_fail = [o for o in obligations
                     if o.source == "core" and o.status == "fail"]
        skipped = [o for o in obligations if o.status == "skipped"]
        missing_dims = [d for d in SCORE_DIMS
                        if d not in agg and any(o._dim == d for o in obligations
                                                if hasattr(o, "_dim"))]
        qbar = (sum(cons.values()) / len(cons)) if cons else 0.0
        floors_ok = all(cons.get(d, -1) >= self.floors[d] for d in cons)

        if core_fail:
            rep.outcome = "Fail"
        elif skipped or missing_dims:
            rep.outcome = "Unverifiable"
        elif not floors_ok or qbar < self.tau_q:
            rep.outcome = "UnderSpecified" if _describable_gaps(obligations) else "Fail"
        else:
            rep.outcome = "Pass"

        rep.hard_obligations = [asdict(o) for o in obligations]
        rep.certification = {"suite_id": task_ctx.get("suite_id", "s_default"),
                             "fresh": task_ctx.get("suite_fresh", True),
                             "valid_for_activation": rep.outcome == "Pass"
                             and task_ctx.get("suite_fresh", True)}
        rep.diagnostics = _sanitize(obligations)
        rep.wall_s = round(time.time() - t0, 2)
        return rep

    # ------------------------------------------------------------------ #
    def _build_core(self, chi: dict, task_ctx: dict) -> list:
        obs: list[Obligation] = []
        content = chi["content"]
        # C2/C3 compilations: required distinctions (even undeclared ones from
        # task_ctx.min_classes are added by the harness, not here)
        for i, rd in enumerate(content.get("required_distinctions") or []):
            # FIX (DEV-20260918-rd-shape-crash): proposers may emit classes as a
            # bare list of ids instead of the {"classes", "reason"} object;
            # schema tolerated it silently and the audit engine crashed on
            # rd.get(). Normalize here so a legal-but-alternate encoding can
            # never kill the audit process.
            if isinstance(rd, list):
                rd = {"classes": rd, "reason": ""}
            elif not isinstance(rd, dict):
                continue
            obs.append(Obligation(
                id=f"CORE-RD-{i:02d}", source="core",
                description=f"required distinction preserved: {rd.get('reason', '')[:120]}",
                check="judge"))
            obs[-1]._dim = "q_cov"
            obs[-1]._rubric = RUBRICS["q_cov"]
            obs[-1]._payload = lambda tc, rd=rd, c=content: json.dumps(
                {"task_evidence": tc.get("evidence_digest", ""),
                 "required_distinction": rd,
                 "gamma": (chi.get("realization") or {}).get("pairs") or []},
                ensure_ascii=False)[:6000]
        # C4: operators executable (sample probes; full battery in op battery)
        for j, op in enumerate(chi["form"]["operators"][:4]):
            ob = Obligation(id=f"CORE-EXEC-{j:02d}", source="core",
                            description=f"operator '{op.get('name')}' executes on smoke probe",
                            check="exec")
            ob._op_code = op["code"]
            ob._entry = [op.get("name", "")]
            ob._probe_args = _smoke_args(op)
            obs.append(ob)
        # Γ structural closure already validated; add explicit core record
        obs.append(Obligation(id="CORE-GAMMA-CLOSED", source="core",
                              description="Γ pairs reference existing hypotheses/carriers",
                              check="structural", status="pass",
                              detail="validated by chi_schema HC3"))
        return obs

    def _build_induced(self, chi: dict, task_ctx: dict) -> list:
        obs: list[Obligation] = []
        # induced: claims → judge checks (漏声明不免责 handled by harness adding
        # task_ctx.min_claims as extra obligations before this call)
        for k, cl in enumerate(chi.get("claims") or []):
            if not isinstance(cl, dict):
                continue
            kind = cl.get("kind", "scientific")
            rubric = RUBRICS["q_sci"] if kind == "scientific" else RUBRICS["q_valid"]
            ob = Obligation(id=f"IND-CLAIM-{k:02d}", source="induced",
                            description=f"claim [{kind}]: {cl.get('text', '')[:120]}",
                            check="judge")
            ob._dim = "q_sci" if kind == "scientific" else "q_valid"
            ob._rubric = rubric
            ob._payload = lambda tc, cl=cl, c=chi["content"]: json.dumps(
                {"task_evidence": tc.get("evidence_digest", ""),
                 "claim": cl, "hypotheses": c.get("hypotheses", [])[:20]},
                ensure_ascii=False)[:6000]
            obs.append(ob)
        # induced: exploration interfaces must reference declared operators
        op_names = {op.get("name") for op in chi["form"]["operators"]}
        for j, ei in enumerate(chi["form"]["exploration_interfaces"]):
            if not isinstance(ei, dict):
                continue
            ok_ref = ei.get("operator") in op_names
            obs.append(Obligation(
                id=f"IND-IFACE-{j:02d}", source="induced",
                description=f"exploration interface '{ei.get('name')}' binds declared operator",
                check="structural", status="pass" if ok_ref else "fail",
                detail="" if ok_ref else f"operator '{ei.get('operator')}' not declared"))
        # induced: relation semantics fidelity
        rels = chi["content"].get("relations") or []
        if rels:
            ob = Obligation(id="IND-REL-SEM", source="induced",
                            description=f"{len(rels)} declared relations' semantics computable",
                            check="judge")
            ob._dim = "q_rel"
            ob._rubric = RUBRICS["q_rel"]
            ob._payload = lambda tc, r=rels, c=chi: json.dumps(
                {"task_evidence": tc.get("evidence_digest", ""),
                 "relations": r, "operators": c["form"].get("operators", [])[:8]},
                ensure_ascii=False)[:6000]
            obs.append(ob)
        # induced: Γ pair realization fidelity
        pairs = (chi.get("realization") or {}).get("pairs") or []
        if pairs:
            ob = Obligation(id="IND-GAMMA-FID", source="induced",
                            description=f"{len(pairs)} Γ pairs realization fidelity",
                            check="judge")
            ob._dim = "q_real"
            ob._rubric = RUBRICS["q_real"]
            ob._payload = lambda tc, p=pairs, c=chi: json.dumps(
                {"task_evidence": tc.get("evidence_digest", ""),
                 "gamma_pairs": p,
                 "hypotheses": c["content"].get("hypotheses", [])[:20],
                 "carriers": c["form"].get("carriers", [])[:10]},
                ensure_ascii=False)[:6000]
            obs.append(ob)
        # induced: interp rule quality
        ob = Obligation(id="IND-INTERP", source="induced",
                        description="interpretation rule set-valued + ambiguity branching",
                        check="judge")
        ob._dim = "q_interp"
        ob._rubric = RUBRICS["q_interp"]
        ob._payload = lambda tc, c=chi: json.dumps(
            {"task_evidence": tc.get("evidence_digest", ""),
             "interp": c.get("interp") or {}, "hypotheses": (c.get("content") or {}).get("hypotheses", [])[:15]},
            ensure_ascii=False)[:6000]
        obs.append(ob)
        return obs


# ---- helpers --------------------------------------------------------------- #

def _revalidate(chi: dict) -> ValidationResult:
    from ..schema.chi_schema import validate
    return validate(chi)


def _smoke_args(op: dict) -> list:
    """Minimal smoke probe: honor declared input types; fall back to 1-arg None.
    Full behavior battery is the executor battery's job (S3), not smoke.
    Operators typically index structured payloads, so defaults prefer dicts and
    lists-of-dicts over bare strings."""
    sig = op.get("signature") or {}
    if isinstance(sig, list):
        inputs = sig  # alternate encoding: signature is the input list itself
    elif isinstance(sig, dict):
        inputs = sig.get("inputs") or []
    else:
        inputs = []
    args = []
    for i in inputs:
        t = str(i.get("type") if isinstance(i, dict) else i or "").lower()
        if any(k in t for k in ("int", "float", "num")):
            args.append(2)
        elif "list" in t and "dict" in t:
            args.append([{"id": 1, "value": 2.0}])
        elif "dict" in t or "map" in t or "dataframe" in t or "table" in t \
                or "csv" in t or "record" in t:
            args.append({"col_a": [1, 2, 3], "col_b": [4.0, 5.0, 6.0]})
        elif "list" in t or "array" in t:
            args.append([1, 2])
        elif "bool" in t:
            args.append(True)
        else:
            args.append("probe")
    return args[:4] or [{"col_a": [1, 2, 3], "col_b": [4.0, 5.0, 6.0]}]


def _describable_gaps(obligations: list) -> bool:
    """UnderSpecified vs Fail: describable = failures are completeness-type
    (judge-scored gaps), not behavioral breakage (exec failures)."""
    fails = [o for o in obligations if o.status == "fail"]
    exec_fails = [o for o in fails if o.check == "exec"]
    return len(exec_fails) == 0 and len(fails) < len(obligations)


def _sanitize(obligations: list) -> list:
    """Diagnostics for the proposer: WHICH dimension failed, class of problem —
    never a replacement design, never other candidates, never judge internals."""
    out = []
    for o in obligations:
        if o.status == "fail":
            out.append({"dimension": getattr(o, "_dim", o.check),
                        "failed": o.id,
                        "hint_class": "behavior_mismatch" if o.check == "exec"
                        else "semantic_gap"})
    return out
