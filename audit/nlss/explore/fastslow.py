"""fastslow.py — Fast/Slow feedback controller (Methods §3; S4 experiment arm).

Arms (S4):
  no_feedback   proposer re-proposes with NO verifier signal at all
  scalar_only   verifier returns only the scalar Q̄ (no dimension detail)
  diagnostic    verifier returns sanitised diagnostics (which dims failed,
                hint_class) — the designed mode; verifier NEVER returns a
                replacement design
  repair_control "verifier-repair" CONTROL arm: a separate LLM is given the
                audit report AND allowed to rewrite the candidate wholesale.
                This arm exists to show the authority split matters: any gain
                it shows is attributed to replacement, not governance.

Adaptive-overfitting readout: each arm runs under reuse-certification and
fresh-certification; we chart ΔQ(reuse) − ΔQ(fresh) per arm. Governance
predicts diagnostic stays informative under fresh certification while
repair_control shows inflated reuse gains (overfit to burned probes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from ..schema.chi_schema import validate
from ..audit.engine import SCORE_DIMS


class FeedbackMode(str, Enum):
    NO_FEEDBACK = "no_feedback"
    SCALAR_ONLY = "scalar_only"
    DIAGNOSTIC = "diagnostic"
    REPAIR_CONTROL = "repair_control"


@dataclass
class Feedback:
    mode: FeedbackMode
    qbar: float | None = None
    diagnostics: list | None = None
    outcome: str | None = None
    repair_draft: str | None = None   # REPAIR_CONTROL only

    def render(self) -> str:
        """The exact text returned to the proposer for its next attempt."""
        if self.mode == FeedbackMode.NO_FEEDBACK:
            return "Previous attempt not accepted. Re-propose."
        if self.mode == FeedbackMode.SCALAR_ONLY:
            return (f"Previous attempt: {self.outcome}. "
                    f"Overall quality Q̄={self.qbar:.3f}. Re-propose.")
        if self.mode == FeedbackMode.DIAGNOSTIC:
            lines = [f"Previous attempt: {self.outcome}. Q̄={self.qbar:.3f}.",
                     "Failed dimensions (no replacement provided):"]
            for d in self.diagnostics or []:
                lines.append(f"- {d['dimension']}: {d['hint_class']} ({d['failed']})")
            return "\n".join(lines)
        if self.mode == FeedbackMode.REPAIR_CONTROL:
            return (f"Previous attempt: {self.outcome}. Audit report follows.\n"
                    "A repair agent has rewritten the candidate; merge its draft:\n"
                    f"{self.repair_draft}")
        raise ValueError(self.mode)


def feedback_from_report(report, mode: FeedbackMode,
                         repair_llm=None) -> Feedback:
    """Build the Feedback object from an AuditReport (or its dict) under the
    given mode."""
    if not isinstance(report, dict):
        report = report.to_dict()
    cons = (report.get("scores") or {}).get("conservative", {}) \
        if isinstance(report.get("scores"), dict) else {}
    qbar = (sum(cons.values()) / len(cons)) if cons else None
    fb = Feedback(mode=mode, qbar=qbar, outcome=report.get("outcome"))
    if mode == FeedbackMode.DIAGNOSTIC:
        fb.diagnostics = report.get("diagnostics", [])
    if mode == FeedbackMode.REPAIR_CONTROL and repair_llm is not None:
        # control arm: hand the FULL report (unsanitised) to an unrestricted LLM
        fb.repair_draft = repair_llm(json.dumps(report, ensure_ascii=False))
    return fb


def adaptive_overfit_delta(scores_reuse: list, scores_fresh: list) -> dict:
    """ΔQ(reuse) − ΔQ(fresh): the S4 overfitting chart's y-value per arm."""
    if not scores_reuse or not scores_fresh:
        return {"delta": None}
    mr = sum(scores_reuse) / len(scores_reuse)
    mf = sum(scores_fresh) / len(scores_fresh)
    return {"delta": round(mr - mf, 4), "q_reuse": round(mr, 4),
            "q_fresh": round(mf, 4)}
