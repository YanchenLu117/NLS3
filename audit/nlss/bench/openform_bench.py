"""openform_bench.py — S1 OpenFormBench lane runner (pilot→full).

Task bank format (JSONL):
  {"task_id", "axis", "semantic_description", "evidence", "required_interactions",
   "min_classes": [{"classes": [[h,h],...], "reason": "evaluator-side floor"}],
   "tolerance": {"T_preserve": 0.05, "T_exec": 1e-6, "max_probe_failures": 0},
   "probe_bank_ref": "probe_banks/<task_family>.jsonl"}

Arms (S1, frozen):
  full            Construct + Hybrid Audit (Fast/Slow diagnostic feedback, 3 attempts)
  fixedmenu_or    candidate forced into M_ref family chosen by EVALUATOR
  fixedmenu_llm   candidate forced into one M_ref family chosen by the LLM itself
  open_noaudit    Construct only, no verifier gate (novelty upper reference)

Metrics: p_nonmenu, novelty@K, accepted, executable (openform_metrics).
Stats: paired sign-permutation (seed 92711) + Holm over one family
       {full_vs_fixedmenu_or, full_vs_fixedmenu_llm, full_vs_open_noaudit,
        fixedmenu_llm_vs_open_noaudit}.

Usage:
  python -m nlss.bench.openform_bench --tasks <jsonl> --arm full \
      --actor $NLSS_LLM_MODEL --decodes 3 --tag pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from ..construct.proposer import Proposer, render_task, check_menu_words
from ..audit.engine import HybridAuditor, SCORE_DIMS
from ..audit.menu_equivalence import judge_nonmenu, compile_required_capabilities
from ..audit.judge_client import JudgeClient
from ..explore.fastslow import FeedbackMode, feedback_from_report
from ..metrics.openform_metrics import arm_summary
from ..schema.op_executor import run_operator

REPO = Path(__file__).resolve().parents[3]
ATTEMPTS = 3          # Fast/Slow loop depth (frozen)
DECODES = 3           # novelty@K (frozen)
ARMS = ["full", "fixedmenu_or", "fixedmenu_llm", "open_noaudit"]


def load_tasks(path: Path) -> list:
    tasks = [json.loads(l) for l in open(path) if l.strip()]
    for t in tasks:
        for k in ("task_id", "semantic_description", "evidence",
                  "required_interactions", "tolerance"):
            if k not in t:
                raise ValueError(f"task {t.get('task_id', '?')} missing {k}")
        # CI: the proposer INSTRUCTION scaffold must be menu-free. Official
        # task evidence text is exempt — domain vocabulary ("graphics design")
        # is task content, not a representation hint; NON_MENU judgment is
        # probe-behavior-only (PREREG §5), surface words never count.
        hits = check_menu_words(render_task({"task_id": t["task_id"]}))
        if hits:
            raise ValueError(f"task {t['task_id']} render scaffold contains menu words: {hits}")
    return tasks


def run_candidate(task: dict, arm: str, actor: str, decode_seed: int,
                  proposer: Proposer, auditor: HybridAuditor,
                  feedback_mode: FeedbackMode = FeedbackMode.DIAGNOSTIC,
                  exp_id: str = "S1A_pilot",
                  per_task_budget: int = 200_000) -> dict:
    """One (task, arm, decode): propose → audit → [fast/slow] → verdict.
    Every proposer/judge call is cost-tracked; per-task budget trips breaker."""
    from .cost_monitor import track_call, task_cost, BudgetExceeded
    ctx = {"suite_id": f"s_{task['task_id']}", "suite_fresh": True,
           "evidence_digest": task["evidence"][:3000]}
    chi, err, meta = None, "init", {}
    fb_text = ""
    report = None
    attempt_trace = []
    for attempt in range(ATTEMPTS):
        task_ctx = dict(task)
        if arm == "fixedmenu_or" or arm == "fixedmenu_llm":
            fam = task.get("_forced_family") if arm == "fixedmenu_or" \
                else task.get("_llm_chosen_family")
            # FIX (DEV-20260917-menu-constraint-dead): the family constraint
            # previously decorated only task_ctx_text (cost-accounting string),
            # while propose() re-rendered the bare task - menu prompts were
            # byte-identical to open prompts. Inject via the key render_task
            # actually consumes so the constraint reaches the prompt.
            task_ctx["required_interactions"] = (
                str(task.get("required_interactions") or "")
                + " (organization constraint: implement the substrate as a " + str(fam) + ")")
        task_ctx_text = render_task(task_ctx)
        if task_cost(exp_id, task["task_id"])["out"] > per_task_budget:
            err = "BudgetExceeded:before_attempt"
            break
        t_prop = time.time()
        chi, err, meta = proposer.propose(
            {**task_ctx, "semantic_description": task_ctx["semantic_description"],
             "evidence": task_ctx["evidence"],
             "required_interactions": task_ctx["required_interactions"]},
            arm=arm, decode_seed=decode_seed + attempt, extra_context=fb_text)
        track_call(exp_id, "proposer", actor,
                   in_tokens=len(task_ctx_text) // 3, out_tokens=len(str(chi)) // 3,
                   latency_s=time.time() - t_prop, task_id=task["task_id"],
                   arm=arm, decode_seed=decode_seed + attempt)
        if chi is None:
            # malformed/failed proposal is an attempt, not a dropped decode:
            # retry within the frozen ATTEMPTS budget (triage-class: parse)
            attempt_trace.append({"attempt": attempt + 1, "stage": "parse", "ok": False})
            fb_text = f"Previous output was invalid ({str(err)[:200]}). Re-propose."
            continue
        if meta.get("schema_fail"):
            attempt_trace.append({"attempt": attempt + 1, "stage": "schema", "ok": False})
            fb_text = f"Schema validation failed: {err}. Fix the structure and re-propose."
            continue
        if arm == "open_noaudit":
            report = _null_report(chi, meta)
            break
        t_j = time.time()
        j0 = dict(auditor.judge.usage)
        # SAFETY NET (DEV-20260918-fuzz-hardening): an engine crash must cost
        # one attempt, never the lane process. Recorded in attempt_trace and
        # a sidecar log for post-hoc triage.
        try:
            report = auditor.audit(chi, ctx)
        except Exception as e:  # noqa: BLE001
            attempt_trace.append({"attempt": attempt + 1, "stage": "audit",
                                  "ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            try:
                with open(REPO / "runs" / "s1" / "audit_engine_errors.log", "a") as ef:
                    ef.write(json.dumps({"exp_id": exp_id, "task_id": task["task_id"],
                                         "arm": arm, "decode_seed": decode_seed,
                                         "attempt": attempt + 1,
                                         "error": f"{type(e).__name__}: {str(e)[:300]}"}) + chr(10))
            except Exception:
                pass
            fb_text = ("Previous attempt failed inside the auditor (" + type(e).__name__ +
                       "). Check every declared structure is a complete object per the schema and re-propose.")
            continue
        ju = auditor.judge.usage
        track_call(exp_id, "judge", auditor.judge.model,
                   in_tokens=ju["in"] - j0["in"], out_tokens=ju["out"] - j0["out"],
                   latency_s=time.time() - t_j, task_id=task["task_id"],
                   arm=arm, decode_seed=decode_seed + attempt)
        attempt_trace.append({"attempt": attempt + 1, "stage": "audit",
                              "outcome": report.outcome, "ok": report.outcome == "Pass"})
        if report.outcome == "Pass":
            break
        try:
            rej = REPO / "runs" / "s1" / "rejected" / f"{exp_id}_{_tk(task)}_{arm}_d{decode_seed}_a{attempt + 1}.json"
            rej.parent.mkdir(parents=True, exist_ok=True)
            rej.write_text(json.dumps(chi, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        fb = feedback_from_report(report, feedback_mode)
        fb_text = fb.render()
    if chi is None:
        return {"task_id": task["task_id"], "arm": arm, "decode_seed": decode_seed,
                "status": f"ERR:{err}", "chi": None, "attempt_trace": attempt_trace}
    out = {"task_id": task["task_id"], "arm": arm, "decode_seed": decode_seed,
           "status": "OK", "candidate_id": chi.get("candidate_id"),
           "substrate_hash": meta.get("substrate_hash", ""),
           "chi": chi, "outcome": report.outcome if report else "Unverifiable",
           "attempt_used": attempt + 1, "attempt_trace": attempt_trace}
    if report is not None:
        out["audit"] = report.to_dict()
        out["all_ops_exec_ok"] = all(
            o.get("status") != "fail"
            for o in out["audit"]["hard_obligations"]
            if o.get("id", "").startswith("CORE-EXEC"))
    else:
        out["all_ops_exec_ok"] = False
    return out


def _null_report(chi: dict, meta: dict):
    """open_noaudit: record structure without gating (novelty reference arm)."""
    class R:
        outcome = "NoAudit"
        def __init__(self):
            from types import SimpleNamespace
            self.scores = {}
            self.diagnostics = []
        def to_dict(self):
            return {"outcome": "NoAudit", "scores": {}, "hard_obligations": [],
                    "diagnostics": [], "substrate_hash": meta.get("substrate_hash", "")}
    return R()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--arms", nargs="+", default=["full"])
    ap.add_argument("--actor", default=os.environ.get("NLSS_LLM_MODEL", ""))
    ap.add_argument("--judge", default=os.environ.get("NLSS_LLM_JUDGE_MODEL", ""))
    ap.add_argument("--decodes", type=int, default=DECODES)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=92711)
    ap.add_argument("--exp-id", default="S1A_pilot")
    args = ap.parse_args()

    def _tk(t):
        return t["task_id"] + "#" + hashlib.md5(
            json.dumps(t, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:8]

    tasks = load_tasks(Path(args.tasks))
    outdir = REPO / "runs" / "s1" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    proposer = Proposer(actor=args.actor)
    auditor = HybridAuditor(judge=JudgeClient(model=args.judge), max_probes=24)
    out_path = outdir / f"s1_{args.actor}_{args.arms[0]}.jsonl"

    # resume semantics: skip (task, arm, decode) rows already OK
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                if r.get("status") == "OK":
                    done.add((r.get("task_key", r["task_id"]), r["arm"], r["decode_seed"]))
            except Exception:
                pass

    for task in tasks:
        for arm in args.arms:
            for d in range(args.decodes):
                if (_tk(task), arm, d) in done:
                    continue
                t0 = time.time()
                row = run_candidate(task, arm, args.actor, d, proposer, auditor, exp_id=args.exp_id)
                row["task_key"] = _tk(task)
                row["wall_s"] = round(time.time() - t0, 1)
                with open(out_path, "a") as g:
                    g.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[s1:{args.tag}] {task['task_id']} {arm} d{d} "
                      f"{row.get('outcome', row.get('status'))} "
                      f"att={row.get('attempt_used')} wall={row['wall_s']}s", flush=True)
    print(f"[s1:{args.tag}] COMPLETE {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
