"""revise_report.py — fixed-revise-round reporting (Fast/Slow curve).

The Fast/Slow loop is frozen at ATTEMPTS=3. For every (task, arm, decode) cell
we already log attempt_used and the per-attempt outcome in the run JSONL. This
module aggregates them into the paper's revise curve:

  revise_curve[arm][k] = share of cells that reached Pass exactly at attempt k
  pass_by_attempt[arm][k] = cumulative pass share by attempt k (survival view)

Report also surfaces the marginal gain of attempt 2/3 (does verifier feedback
add value?) and its cost (extra proposer+judge calls from cost_monitor).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def revise_curve(rows: list, max_attempts: int = 3) -> dict:
    by_arm = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    pass_by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.get("status") != "OK":
            continue
        arm = r.get("arm", "?")
        att = min(int(r.get("attempt_used", 1)), max_attempts)
        totals[arm] += 1
        passed = r.get("outcome") == "Pass"
        by_arm[arm][att] += 1 if passed else 0
        for k in range(att, max_attempts + 1):
            if passed:
                pass_by[arm][k] += 1
    out = {}
    for arm, tot in totals.items():
        if not tot:
            continue
        exact = {k: round(by_arm[arm].get(k, 0) / tot, 4) for k in range(1, max_attempts + 1)}
        cum = {k: round(pass_by[arm].get(k, 0) / tot, 4) for k in range(1, max_attempts + 1)}
        out[arm] = {"n": tot, "pass_exact_at": exact, "pass_cum_by": cum,
                    "marginal_gain_att2": round(cum.get(2, 0) - cum.get(1, 0), 4),
                    "marginal_gain_att3": round(cum.get(3, 0) - cum.get(2, 0), 4)}
    return out


def report_from_file(path: Path, max_attempts: int = 3) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return revise_curve(rows, max_attempts)


def write_report(exp_id: str, curve: dict, cost_summary: dict | None = None,
                 out_dir: Path = None) -> Path:
    out_dir = out_dir or (REPO / "paper_lab" / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{exp_id}_revise.md"
    lines = [f"# Revise report — {exp_id} (fixed ATTEMPTS=3)", ""]
    for arm, c in sorted(curve.items()):
        lines.append(f"## {arm} (n={c['n']})")
        lines.append(f"- pass exactly at attempt: {c['pass_exact_at']}")
        lines.append(f"- cumulative pass by attempt: {c['pass_cum_by']}")
        lines.append(f"- marginal gain: att2 {c['marginal_gain_att2']}, att3 {c['marginal_gain_att3']}")
        lines.append("")
    if cost_summary:
        lines.append("## cost per arm (from cost_monitor)")
        for arm, v in sorted(cost_summary.items()):
            lines.append(f"- {arm}: calls={v['calls']} in={v['in']} out={v['out']} "
                         f"lat_p95={v.get('latency_p95_s')}s")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
