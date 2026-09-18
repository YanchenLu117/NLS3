"""V8 — §13 fairness caps, Pareto rule, and §17/§18 completion checklist.

Equal-resource controlled comparison (Detail §13.1): seven frozen caps; NLSS's
full chain (induction, audit, readout, extraction, controller, revision)
consumes them; **if Full cannot complete under the cap, the incomplete result
IS the primary equal-resource result**.  Pareto rule (§13.3): no worse = the
lower paired 95% CI bound exceeds −δ_harm; strictly better = the lower bound
exceeds δ_min; consistent domination requires it on every preregistered
compatible campaign AND at least two campaigns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

CAP_KEYS: tuple[str, ...] = (
    "llm_calls",
    "generated_tokens",
    "persistent_state_tokens",
    "tool_calls",
    "oracle_budget",
    "wall_time_seconds",
    "retries",
)


class CapExceeded(RuntimeError):
    """A cap was exceeded: the incomplete result remains the primary
    equal-resource result (Detail §13.1) — flagged, not silently extended."""


@dataclass
class ResourceMeter:
    """Equal-resource meter: counts consumption against the frozen caps."""

    caps: Mapping[str, float]

    def __post_init__(self) -> None:
        unknown = set(self.caps) - set(CAP_KEYS)
        if unknown:
            raise ValueError(f"unknown cap keys: {', '.join(sorted(unknown))} (§13.1 defines {CAP_KEYS})")
        self.used: dict[str, float] = {k: 0.0 for k in CAP_KEYS}
        self.exceeded: dict[str, bool] = {k: False for k in CAP_KEYS}

    def consume(self, key: str, amount: float) -> None:
        if key not in CAP_KEYS:
            raise ValueError(f"unknown resource key: {key}")
        if self.exceeded[key]:
            return  # already over cap: further consumption is off-ledger noise
        self.used[key] += float(amount)
        if self.used[key] > float(self.caps[key]):
            self.exceeded[key] = True

    def within(self, key: str) -> bool:
        return not self.exceeded[key]

    @property
    def incomplete_under_cap(self) -> bool:
        return any(self.exceeded.values())

    def usage(self) -> dict[str, float]:
        return dict(self.used)


def pareto_compare(
    *,
    ci_low: float,
    delta_harm: float,
    delta_min: float,
) -> str:
    """Single-axis verdict (Detail §13.3): positive favors method A.

    Returns "no_worse_strictly_better" | "no_worse" | "dominated".
    """
    if ci_low > -delta_harm:
        if ci_low > delta_min:
            return "strictly_better"
        return "no_worse"
    return "dominated"


def consistent_domination(
    axis_verdicts_by_campaign: Mapping[str, Mapping[str, str]],
    *,
    utility_axes: Sequence[str],
    resource_axes: Sequence[str],
) -> bool:
    """§13.3: A consistently Pareto-dominates B iff on EVERY preregistered
    compatible campaign no axis is dominated AND at least one utility/resource
    axis is strictly better — and there are at least two campaigns
    (fewer → S1 descriptive only, return False)."""
    campaigns = list(axis_verdicts_by_campaign)
    if len(campaigns) < 2:
        return False
    for campaign in campaigns:
        axes = axis_verdicts_by_campaign[campaign]
        for axis in (*utility_axes, *resource_axes):
            if axes.get(axis) == "dominated":
                return False
        if not any(axes.get(a) == "strictly_better" for a in (*utility_axes, *resource_axes)):
            return False
    return True


# ------------------------------------------------- §17/§18 completion gate

CompletionCheck = Callable[[Mapping[str, Any]], bool]

CHECKLIST: tuple[tuple[str, str], ...] = (
    ("terminal_status_all_units", "all planned units have a terminal status"),
    ("failures_represented", "all failures remain represented"),
    ("evaluator_isolation_hashes", "evaluator isolation and artifact hashes pass"),
    ("primary_contrasts_complete", "every primary contrast has a longitudinal result and resource ledger"),
    ("certification_language_matches", "certification language matches the achieved level"),
    ("no_post_freeze_changes", "no post-freeze result-dependent change is unresolved"),
    ("identity_ledgers_pass", "all baseline identity and human-intervention ledgers pass"),
    ("claim_table_updated", "the claim table has been updated per preregistered failure rules"),
)


@dataclass
class CompletionReport:
    checks: dict[str, bool] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def run_completion_checklist(context: Mapping[str, Any]) -> CompletionReport:
    """§18 phase-completion assertions.  ``context`` must carry a boolean per
    checklist key; a missing key is a failed check (never silently passed)."""
    report = CompletionReport()
    for key, _description in CHECKLIST:
        value = context.get(key)
        passed = value is True
        report.checks[key] = passed
        if not passed:
            problems = context.get(f"{key}_problems", f"{key} not satisfied")
            report.problems.append(f"{key}: {problems}")
    return report
