"""failure_triage.py — automatic failure classification + bounded retry + report.

After each run (or periodically), scan a run's JSONL rows and classify failures:

  schema_fail      proposer output failed χ validation (method-side; NO auto
                   retry — goes to root-cause table)
  judge_parse      judge returned unparseable JSON (infra-side; auto retry
                   bounded, glmretry protocol)
  sandbox_timeout  operator exceeded 30s in jail (method-side; no auto retry,
                   but recorded as fidelity signal)
  budget           BudgetExceeded (policy-side; no retry, cell marked)
  infra_5xx        gateway/proxy 5xx or timeouts (infra-side; auto retry
                   bounded, exponential backoff)

The triage report lands in paper_lab/reports/<exp_id>_triage.md with per-class
counts, example errors, and the retry actions taken. Root-cause table is the
input to the postmortem loop (v9 discipline: 取证定位 before any fix).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

CLASSES = ["schema_fail", "judge_parse", "sandbox_timeout", "budget",
           "infra_5xx", "empty_output", "unknown"]

_PATTERNS = [
    ("budget", re.compile(r"BudgetExceeded|budget violation", re.I)),
    ("schema_fail", re.compile(r"schema_fail|HC[0-7]|invalid JSON|SyntaxError", re.I)),
    ("judge_parse", re.compile(r"judge parse fail|bad judge JSON|unparseable judge", re.I)),
    ("sandbox_timeout", re.compile(r"timeout.*jail|jail exec failed|TimeoutExpired", re.I)),
    ("infra_5xx", re.compile(r"502|503|504|ConnectionError|timeout(?!.*jail)|retries", re.I)),
    ("empty_output", re.compile(r"empty content|no JSON object|no callable entry", re.I)),
]


def classify(err: str) -> str:
    if not err:
        return "unknown"
    for cls, pat in _PATTERNS:
        if pat.search(err):
            return cls
    return "unknown"


def triage_rows(rows: list) -> dict:
    ok = [r for r in rows if r.get("status") == "OK"]
    fail = [r for r in rows if r.get("status", "").startswith("ERR")]
    by_class = Counter()
    examples = defaultdict(list)
    for r in fail:
        c = classify(r.get("status", ""))
        by_class[c] += 1
        if len(examples[c]) < 3:
            examples[c].append(r.get("task_id"), ) if False else examples[c].append(
                {"task": r.get("task_id"), "err": r.get("status", "")[:200]})
    return {"n_total": len(rows), "n_ok": len(ok), "n_fail": len(fail),
            "by_class": dict(by_class),
            "examples": {k: v for k, v in examples.items()},
            "auto_retry_eligible": {c: by_class[c] for c in ("infra_5xx", "judge_parse")
                                    if by_class[c]}}


def triage_file(path: Path) -> dict:
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return triage_rows(rows)


def write_report(exp_id: str, tri: dict, out_dir: Path = None) -> Path:
    out_dir = out_dir or (REPO / "paper_lab" / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{exp_id}_triage.md"
    lines = [f"# Triage — {exp_id}", "",
             f"- cells: {tri['n_total']} | ok: {tri['n_ok']} | fail: {tri['n_fail']}", "",
             "## by class", ""]
    for c, n in sorted(tri["by_class"].items()):
        lines.append(f"- `{c}`: {n}")
    lines += ["", "## auto-retry eligible (bounded, next pass)", ""]
    for c, n in tri["auto_retry_eligible"].items():
        lines.append(f"- `{c}`: {n} cells → retry pass scheduled")
    lines += ["", "## examples", ""]
    for c, exs in tri["examples"].items():
        for e in exs:
            lines.append(f"- `{c}` {e['task']}: {e['err'][:160]}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
